"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import math
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from U_NET_Token_Masked import AudioEncoder, ChannelLayerNorm

# AudioEncoder total stride (4 * 4 * 4). Keep in sync with that class.
AUDIO_HOP = 64


# ----------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------

def build_onset_frame_target(
    onsets_sec: Sequence[float],
    n_frames: int,
    window_sec: float,
    sigma_frames: float = 1.0,
) -> torch.Tensor:
    """Gaussian-smeared frame target.
    Returns (n_frames,) float in [0, 1].
    """
    out = torch.zeros(n_frames, dtype=torch.float32)
    if len(onsets_sec) == 0 or window_sec <= 0:
        return out

    centers = torch.tensor(
        [float(o) / window_sec * n_frames for o in onsets_sec], dtype=torch.float32
    )
    centers = centers[(centers >= -3 * sigma_frames) & (centers <= n_frames + 3 * sigma_frames)]
    if centers.numel() == 0:
        return out

    idx = torch.arange(n_frames, dtype=torch.float32).unsqueeze(1)      # (F, 1)
    d = idx - centers.unsqueeze(0)                                       # (F, K)
    g = torch.exp(-0.5 * (d / max(sigma_frames, 1e-6)) ** 2)
    return g.max(dim=1).values.clamp_(0.0, 1.0)


def frame_validity_mask(effective_samples: int, n_frames: int, hop: int = AUDIO_HOP) -> torch.Tensor:

    m = torch.zeros(n_frames, dtype=torch.float32)
    n_valid = int(math.ceil(max(effective_samples, 1) / hop))
    m[: min(n_valid, n_frames)] = 1.0
    return m


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class _DilatedBlock(nn.Module):

    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = ChannelLayerNorm(ch)
        self.conv1 = nn.Conv1d(ch, ch * 2, kernel, padding=dilation * (kernel // 2), dilation=dilation)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(ch * 2, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(self.norm(x))
        h = self.drop(self.conv2(self.act(h)))
        return x + h


class OnsetHead(nn.Module):
    """Raw audio + spectral features -> per-frame onset logits.

    Parameters
    ----------
    audio_embed_dim : channels of the internal AudioEncoder.
    inject_feature_dim : channel count of `cond` as it is actually passed in.
    dilations : receptive field of the trunk. (1, 2, 4, 8) at 4 ms/frame gives
        ~120 ms of context each side, about right for a plucked attack.
    count_cond_classes : if set, the head is FiLM-conditioned on the event count
    """

    def __init__(
        self,
        audio_embed_dim: int = 64,
        inject_feature_dim: int = 514,
        hidden: int = 128,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        count_cond_classes: Optional[int] = None,
    ):
        super().__init__()
        self.audio_embed_dim = audio_embed_dim
        self.inject_feature_dim = inject_feature_dim
        self.hidden = hidden
        self.count_cond_classes = count_cond_classes

        self.audio_encoder = AudioEncoder(audio_embed_dim, dropout=dropout)
        self.c_conv = nn.Conv1d(inject_feature_dim, hidden, 3, padding=1)
        self.fuse = nn.Conv1d(audio_embed_dim + hidden, hidden, 1)

        self.trunk = nn.ModuleList(
            [_DilatedBlock(hidden, d, dropout=dropout) for d in dilations]
        )

        if count_cond_classes:
            self.count_film = nn.Sequential(
                nn.Linear(count_cond_classes, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden * 2),
            )
            nn.init.zeros_(self.count_film[-1].weight)
            nn.init.zeros_(self.count_film[-1].bias)

        self.norm_out = ChannelLayerNorm(hidden)
        self.to_logits = nn.Conv1d(hidden, 1, 1)

        nn.init.normal_(self.to_logits.weight, std=0.01)
        nn.init.constant_(self.to_logits.bias, -4.0)

    def forward(self, audio: torch.Tensor, cond: torch.Tensor,
                count_dist: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        audio      : (B, T_samples, 1) raw waveform, same tensor the U-Net gets
        cond       : (B, L_spec, inject_feature_dim) spectral features
        count_dist : (B, count_cond_classes) softmax over event counts from the
                     count head, or a one-hot ground truth. Required when
                     count_cond_classes was set; ignored otherwise.

        returns
            logits   : (B, L_frames) onset logits, L_frames = T_samples // 64
            features : (B, hidden, L_frames) trunk activations (exposed for
                       logging / future conditioning; unused by default)
        """
        a = self.audio_encoder(audio)                       # (B, A, L_frames)
        c = self.c_conv(cond.permute(0, 2, 1))              # (B, H, L_spec)

        if c.shape[-1] != a.shape[-1]:
            if c.shape[-1] > a.shape[-1]:
                c = F.adaptive_avg_pool1d(c, a.shape[-1])
            else:
                c = F.interpolate(c, size=a.shape[-1], mode="linear", align_corners=False)

        h = self.fuse(torch.cat([a, c], dim=1))

        if self.count_cond_classes:
            if count_dist is None:
                raise ValueError("OnsetHead was built with count_cond_classes; "
                                 "pass count_dist")
            gamma, beta = self.count_film(count_dist.float()).chunk(2, dim=-1)
            h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        for block in self.trunk:
            h = block(h)

        logits = self.to_logits(self.norm_out(h)).squeeze(1)  # (B, L_frames)
        return logits, h


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def onset_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked BCE with positive-class reweighting.
    """
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)

    loss = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none", pos_weight=pos_weight
    )
    if frame_mask is None:
        return loss.mean()

    frame_mask = frame_mask.to(loss.dtype)
    return (loss * frame_mask).sum() / frame_mask.sum().clamp(min=1.0)


@torch.no_grad()
def compute_onset_pos_weight(dataloader, key="onset_frames", mask_key="onset_frame_mask",
                             cap: float = 50.0) -> torch.Tensor:
    pos, tot = 0.0, 0.0
    for batch in dataloader:
        t = batch[key].float()
        m = batch[mask_key].float() if mask_key in batch else torch.ones_like(t)
        pos += float((t * m).sum())
        tot += float(m.sum())
    neg = max(tot - pos, 1.0)
    return torch.tensor(min(neg / max(pos, 1.0), cap), dtype=torch.float32)


# ----------------------------------------------------------------------
# Decoding: posteriorgram -> per-slot onset times
# ----------------------------------------------------------------------

@torch.no_grad()
def decode_onset_times(
    logits: torch.Tensor,
    max_events: int,
    n_events: Optional[torch.Tensor] = None,
    threshold: float = 0.3,
    min_distance: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Peak-pick a posteriorgram into per-event-slot times.

    logits     : (B, L) frame logits
    max_events : number of event slots to fill (dataset.max_events)
    n_events   : (B,) how many peaks to keep per item. Pass the count head's
                 prediction — that makes the decoder hyperparameter-free and
                 keeps the two heads consistent. If None, thresholding decides.
    min_distance : peaks closer than this many frames are suppressed
                 (2 frames = 8 ms; guitar events are never closer).

    returns
        times : (B, max_events) float in [0, 1], sorted ascending, 0 where invalid
        mask  : (B, max_events) bool, True for slots that carry a real onset
    """
    prob = torch.sigmoid(logits)
    B, L = prob.shape
    device = prob.device

    ramp = torch.linspace(1.0, 0.0, L, device=device, dtype=torch.float64) * 1e-6
    prob_tb = prob.double() + ramp
    k = 2 * min_distance + 1
    pooled = F.max_pool1d(prob_tb.unsqueeze(1), kernel_size=k, stride=1, padding=min_distance).squeeze(1)
    is_peak = prob_tb >= pooled
    score = torch.where(is_peak, prob, torch.zeros_like(prob))

    if n_events is None:
        score = torch.where(score > threshold, score, torch.zeros_like(score))
        n_keep = (score > 0).sum(dim=1)
    else:
        n_keep = n_events.to(device).long()
    n_keep = n_keep.clamp(min=1, max=max_events)

    k_top = min(max_events, L)
    top_val, top_idx = score.topk(k_top, dim=1)                     # (B, k_top)
    if k_top < max_events:                                          # L < max_events
        pad = max_events - k_top
        top_val = F.pad(top_val, (0, pad), value=0.0)
        top_idx = F.pad(top_idx, (0, pad), value=0)

    rank = torch.arange(max_events, device=device).unsqueeze(0)      # (1, E)
    keep = (rank < n_keep.unsqueeze(1)) & (top_val > 0)

    times = top_idx.float() / max(L - 1, 1)

    times_sorted, order = torch.where(keep, times, torch.full_like(times, 2.0)).sort(dim=1)
    keep_sorted = torch.gather(keep, 1, order)
    times_out = torch.where(keep_sorted, times_sorted, torch.zeros_like(times_sorted))

    return times_out, keep_sorted

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

@torch.no_grad()
def onset_metrics(
    pred_times: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_times: torch.Tensor,
    gt_mask: torch.Tensor,
    window_sec: float = 1.0,
    tolerance_sec: float = 0.05,
) -> dict:

    tol = tolerance_sec / max(window_sec, 1e-8)
    tp = fp = fn = 0
    abs_errs: List[float] = []

    B = pred_times.shape[0]
    for b in range(B):
        p = pred_times[b][pred_mask[b]].tolist()
        g = gt_times[b][gt_mask[b]].tolist()
        used = [False] * len(g)
        for pt in sorted(p):
            best, best_d = -1, tol
            for j, gt in enumerate(g):
                if used[j]:
                    continue
                d = abs(pt - gt)
                if d <= best_d:
                    best, best_d = j, d
            if best >= 0:
                used[best] = True
                tp += 1
                abs_errs.append(best_d * window_sec)
            else:
                fp += 1
        fn += used.count(False)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "mae_sec": (sum(abs_errs) / len(abs_errs)) if abs_errs else float("nan"),
    }


def print_onset_metrics(m: dict, save_path: Optional[str] = None, prefix: str = "") -> str:
    txt = (
        f"{prefix} onset  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
        f"TP={m['tp']} FP={m['fp']} FN={m['fn']}  MAE={m['mae_sec'] * 1000:.1f} ms"
    )
    print(txt)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    return txt