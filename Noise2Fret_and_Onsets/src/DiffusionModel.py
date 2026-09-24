"""
Created on Tue Nov 2 08:14:08 2025

@author: Riccardo Simionato

"""

from math import pi
from torch import Tensor
import torch.nn as nn
from typing import Tuple
from DiffusionUtils import UniformDistribution, extend_dim, LinearSchedule
import torch
from utils import set_trainable
from tqdm import tqdm
from einops import repeat
import torch.nn.functional as F
from AuxiliaryLoss import (pc_tokens_to_binary, fret_distance, cof_chord_distance,
                           jaccard_tonal_distance, string_activity_jaccard_loss_soft,
                           hand_span_penalty_soft, soft_fret_expectation, pc_probs_to_soft_binary)
from count_loss_utils import build_neighbor_smoothed_targets, soft_cross_entropy
from OnsetsHead import onset_bce_loss, decode_onset_times


N_STRINGS = 6
PAD_FRET = -1
PAD_PC = -1
OPEN_PITCHES = [64, 59, 55, 50, 45, 40]  # s1=E4(high) ... s6=E2(low), matching GuitarPro


def event_mask_from_tab_mask(tab_mask: torch.Tensor) -> torch.Tensor:
    """
    tab_mask : (B, T*6) bool, True at PAD (per string slot)
    returns  : (B, 1, T) float, 1.0 where the event has at least one real string
    """
    B = tab_mask.shape[0]
    real = (~tab_mask).view(B, tab_mask.shape[1] // N_STRINGS, N_STRINGS).any(dim=-1)
    return real.float().unsqueeze(1)  # (B, 1, T)


class DiffusionModel(nn.Module):

    STAGES = ("unet", "count", "onset")
    STAGE_COMPONENTS = {
        "unet": ("model", "embeddings"),
        "count": ("count_head",),
        "onset": ("onset_head",),
    }

    def __init__(self, model, count_head, onset_head, noise_steps=100, embed_dim=32, vocab_size=10,
                 stage="unet", count_smoothing=0.1, count_class_weights=None, zero_events_impossible=True,
                 onset_pos_weight=None, gt_onset_prob=1.0, onset_count_source="gt", force_gt_onsets=True,
                 onset_threshold=0.3, onset_min_distance=2,
                 count_source_for_onset="gt", gt_count_prob=1.0,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        super().__init__()
        self.model = model.to(device)
        self.count_head = count_head.to(device)
        self.onset_head = onset_head.to(device)
        self.noise_steps = noise_steps
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.device = device
        self.max_grad_norm = 1.0  # Prevent exploding gradients
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.use_gt_count = True

        assert stage in self.STAGES, f"stage must be one of {self.STAGES}, got {stage!r}"
        self.stage = stage
        self._apply_stage()

        # --- count-head loss config ---
        self.count_smoothing = count_smoothing  # 0.0 disables neighbor smoothing
        self.zero_events_impossible = zero_events_impossible
        if count_class_weights is not None:
            self.register_buffer("count_class_weights", count_class_weights.float())
        else:
            self.count_class_weights = None

        # --- onset-head config ---
        if onset_pos_weight is not None:
            self.register_buffer("onset_pos_weight", torch.as_tensor(onset_pos_weight).float())
        else:
            self.onset_pos_weight = None
        self.gt_onset_prob = float(gt_onset_prob)
        self.force_gt_onsets = bool(force_gt_onsets)

        self.onset_count_source = onset_count_source
        self.onset_threshold = onset_threshold
        self.onset_min_distance = onset_min_distance

        assert count_source_for_onset in ("gt", "pred", "mix"), count_source_for_onset
        self.count_source_for_onset = count_source_for_onset
        self.gt_count_prob = float(gt_count_prob)
    # ------------------------------------------------------------------
    # Stage handling
    # ------------------------------------------------------------------

    def _apply_stage(self):
        set_trainable(self.model, self.stage == "unet")
        set_trainable(self.embeddings, self.stage == "unet")
        set_trainable(self.count_head, self.stage == "count")
        set_trainable(self.onset_head, self.stage == "onset")
        self.train(self.training)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.model.train(mode and self.stage == "unet")
        self.embeddings.train(mode and self.stage == "unet")
        if self.count_head is not None:
            self.count_head.train(mode and self.stage == "count")
        if self.onset_head is not None:
            self.onset_head.train(mode and self.stage == "onset")
        return self

    def set_stage(self, stage):
        assert stage in self.STAGES
        self.stage = stage
        self._apply_stage()

    def active_module(self) -> nn.Module:
        if self.stage == "unet":
            return nn.ModuleDict({"model": self.model, "embeddings": self.embeddings})
        if self.stage == "count":
            return self.count_head
        return self.onset_head

    def component(self, name: str) -> nn.Module:
        return getattr(self, name)

    @torch.no_grad()
    def component_fingerprint(self):
        out = {}
        for name in ("model", "embeddings", "count_head", "onset_head"):
            mod = getattr(self, name, None)
            if mod is None:
                continue
            param_names = {k for k, _ in mod.named_parameters()}
            p_sum, b_sum = 0.0, 0.0
            for k, v in mod.state_dict().items():
                s = float(v.detach().double().abs().sum()) if v.is_floating_point() \
                    else float(v.detach().double().sum())
                if k in param_names:
                    p_sum += s
                else:
                    b_sum += s
            out[name] = (p_sum, b_sum)
        return out


    def set_gt_onset_prob(self, p):
        self.gt_onset_prob = float(min(max(p, 0.0), 1.0))

    def set_gt_count_prob(self, p):
        self.gt_count_prob = float(min(max(p, 0.0), 1.0))

    @property
    def use_gt_count(self) -> bool:
        return self.count_source_for_onset == "gt"

    @use_gt_count.setter
    def use_gt_count(self, value: bool):
        self.count_source_for_onset = "gt" if value else "pred"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T, 6, n_classes) float one-hot  OR  (B, T, 6) integer IDs
        out : (B, T, 6*embed_dim)  continuous embedding ready for diffusion
        """
        B, T = x.shape[:2]
        ids = x.long()  # already integer IDs
        emb = self.embeddings(ids.view(B, T // N_STRINGS, -1))  # (B, T, embed_dim)
        B, T = emb.shape[:2]

        return emb.view(B, T, N_STRINGS * self.embed_dim)  # (B, T, 6*embed_dim)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, T, 6*embed_dim)  denoised latent
        out  : (B, T, 6, n_classes) logits — argmax(-1) gives class IDs
        """
        B, T = x.shape[:2]
        x_str = x.view(B, T, N_STRINGS, self.embed_dim)  # (B, T, 6, E)
        W = self.embeddings.weight  # (n_classes, E)
        logits = x_str @ W.T  # (B, T, 6, n_classes)
        return logits

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def noise_audios(self, x, sigmas_batch):
        # Get noise
        noise = torch.randn_like(x)

        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        return x_noisy, noise, v_target

    def sample_timesteps(self, batch_size, device, dim):
        """Sample random timesteps."""
        sigmas = UniformDistribution()(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch

    def sample_timesteps_val(self, batch_size, device, dim):
        """Sample validation timesteps."""
        sigmas = torch.linspace(0.0, 1.0, batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _rounding_loss(self, x0_pred: torch.Tensor, target_ids: torch.Tensor, tab_mask: torch.Tensor) -> torch.Tensor:
        """
        x0_pred    : (B, T, 6*embed_dim)
        target_ids : (B, T, 6)  integer class IDs
        """

        logits = self.decode(x0_pred)  # (B, T, 6, n_classes)

        # CE expects (N, C) and (N,)
        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),  # (B*T*6, n_classes)
            target_ids.reshape(-1),  # (B*T*6,)
            # reduction='none',                                  # (B*T*6,) per-position
        )
        use_real = False
        if use_real:
            real = (~tab_mask).reshape(-1).float()
            return (ce * real).sum() / real.sum().clamp(min=1)
        return ce

    def avg_acc(self, x0_pred: torch.Tensor, target_ids: torch.Tensor, tab_mask: torch.Tensor) -> float:

        logits = self.decode(x0_pred)
        pred_ids = torch.argmax(logits, dim=-1)

        b, t, s = pred_ids.shape
        target_ids = target_ids.view(b, t, s)

        # tab_mask is flattened and True only for PAD positions
        valid = (~tab_mask).view_as(pred_ids)

        # Select only non-padding string positions
        correct = pred_ids[valid] == target_ids[valid]
        return correct.float().mean().item()

    @staticmethod
    def _build_pc_frets(tab: torch.Tensor):
        """
        Derive frets and pitch-class vectors from a (6, 21) one-hot tab matrix.

        Returns
        -------
        frets : torch.Tensor, shape (6,)   int64   -1=muted, 0=open, 1..19=fret
        pc    : torch.Tensor, shape (6,)   int64   -1=muted, 0..11=pitch class
        """
        frets = torch.full((tab.shape[0], N_STRINGS,), PAD_FRET, dtype=torch.long)
        pc = torch.full((tab.shape[0], N_STRINGS,), PAD_PC, dtype=torch.long)

        classes = tab
        if len(tab.shape) == 4:
            classes = tab.argmax(dim=-1)  # (6,)  — class index per string
        for t in range(tab.shape[0]):
            for s in range(N_STRINGS):
                cls = int(classes[t, s].item())
                if cls == 0:
                    # muted — keep PAD values
                    continue
                fret = cls - 1  # class 1 → fret 0 (open), class 2 → fret 1, …
                frets[t, s] = fret
                pc[t, s] = (OPEN_PITCHES[s] + fret) % 12

        return frets, pc

    # ------------------------------------------------------------------
    # Onset conditioning
    # ------------------------------------------------------------------

    def _count_dist(self, count_logits):
        if count_logits is None:
            return None
        return torch.softmax(count_logits.detach(), dim=-1)

    def _gt_count_dist(self, count_target, num_classes):
        """One-hot distribution from ground-truth class INDICES, (B,) long."""
        if count_target is None:
            return None
        idx = count_target.long().clamp_(0, num_classes - 1)
        return F.one_hot(idx, num_classes).float()

    def _onset_count_dist(self, count_logits, count_target):
        if count_logits is None:
            return None
        src = self.count_source_for_onset
        if src == "pred":
            return self._count_dist(count_logits)
        if src == "gt":
            return self._gt_count_dist(count_target, count_logits.shape[-1])
        # "mix": teacher forcing, decided per batch
        if self.training and float(torch.rand(())) < self.gt_count_prob:
            return self._gt_count_dist(count_target, count_logits.shape[-1])
        return self._count_dist(count_logits)

    def _count_forward(self, audio, spec):

        if self.count_head is None:
            return None
        if self.stage == "count":
            return self.count_head(audio, spec)
        with torch.no_grad():
            return self.count_head(audio, spec).detach()

    def _onset_forward(self, audio, spec, count_dist=None):

        if self.onset_head is None:
            return None
        if self.stage == "onset":
            logits, _feat = self.onset_head(audio, spec, count_dist=count_dist)
            return logits
        with torch.no_grad():
            logits, _feat = self.onset_head(audio, spec, count_dist=count_dist)
        return logits.detach()

    def _use_gt_onsets(self):
        return self.force_gt_onsets or (
                self.training
                and self.gt_onset_prob > 0.0
                and float(torch.rand(())) < self.gt_onset_prob
        )

    def _n_events(self, n_real, count_logits):
        if self.onset_count_source == "gt" and n_real is not None:
            return n_real.long()
        if self.onset_count_source == "count_head" and count_logits is not None:
            n = torch.argmax(count_logits, dim=-1)
            return n + 1 if self.zero_events_impossible else n
        return None


    def _onset_cond(self, onset_logits, n_slots, n_real, count_logits,
                    onset_frames=None, onset_times=None, onset_times_mask=None):
        use_gt = onset_times is not None and self._use_gt_onsets()
        n_events = self._n_events(n_real, count_logits)

        if use_gt:
            times = onset_times[:, :n_slots]
            valid = onset_times_mask[:, :n_slots].bool()
            prob = onset_frames.float() if onset_frames is not None else None
            if n_real is not None:
                n_events = n_real.long()  # GT conditioning implies GT count
        elif onset_logits is not None:
            prob = torch.sigmoid(onset_logits)
            times, valid = decode_onset_times(
                onset_logits, max_events=n_slots, n_events=n_events,
                threshold=self.onset_threshold, min_distance=self.onset_min_distance,
            )
        else:
            return None

        if n_events is None:
            n_events = valid.sum(dim=1)

        return {"prob": prob, "n_events": n_events, "times": times,
                "valid": valid, "n_slots": n_slots}


    def _forward_pass(self, target_ids, audio, cond, sigmas_fn, tab_mask,
                      onset_frames=None, onset_frame_mask=None,
                      onset_times=None, onset_times_mask=None):

        target_emb = self.encode(target_ids)  # (B, T, 6*E)

        sigmas_t, sigmas_batch_t = sigmas_fn(
            target_emb.shape[0], self.device, target_emb.ndim
        )

        x_t, _, v_target = self.noise_audios(target_emb, sigmas_batch_t)

        mask = event_mask_from_tab_mask(tab_mask)  # (B, 1, T) float, 1.0 = real event

        # --- onset head + conditioning ---------------------------------
        spec = cond[..., :-1]
        n_real = (~tab_mask).view(tab_mask.shape[0], -1).sum(dim=1) // 6  # (B,)
        count_logits = self.count_head(audio, spec)
        count_target = (n_real - 1).long() if self.zero_events_impossible else n_real.long()

        count_dist = self._onset_count_dist(count_logits, count_target)
        onset_logits = self._onset_forward(audio, spec, count_dist)

        if self.count_smoothing > 0:
            target_dist = build_neighbor_smoothed_targets(
                count_target, num_classes=count_logits.shape[-1], smoothing=self.count_smoothing
            )
            count_loss = soft_cross_entropy(
                count_logits, target_dist,
                target_hard=count_target, class_weights=self.count_class_weights,
            )
        elif self.count_class_weights is not None:
            count_loss = F.cross_entropy(count_logits, count_target,
                                         weight=self.count_class_weights.to(count_logits.device))
        else:
            count_loss = F.cross_entropy(count_logits, count_target)

        onset_cond = self._onset_cond(
            onset_logits, target_emb.shape[1], n_real, count_logits,
            onset_frames, onset_times, onset_times_mask,
        )

        predicted_v = self.model(
            x_t, sigmas_t, audio, cond, mask,
            onset_cond=onset_cond
        )
        alphas_b, betas_b = self.get_alpha_beta(sigmas_batch_t)
        x0_pred = alphas_b * x_t - betas_b * predicted_v  # (B, T, 6*E)
        t_mean = sigmas_batch_t.mean()  # curriculum weighting

        use_real = False # Debug
        if use_real:
            # tab_mask (B T*6)
            real = (~tab_mask).view(tab_mask.shape[0], tab_mask.shape[1] // 6, -1).any(dim=-1, keepdim=True)
            real = real.expand_as(x0_pred)
            loss = F.mse_loss(
                x0_pred[real],
                target_emb[real],
            )
            # loss = F.mse_loss(
            #     predicted_v[real],
            #     v_target[real],
            # )
        else:
            loss = F.mse_loss(
                    x0_pred,
                    target_emb,
                )

            # loss = F.mse_loss(
            #     predicted_v,
            #     v_target,
            # )

        rounding_loss = self._rounding_loss(x0_pred, target_ids, tab_mask)
        loss = loss + 0.1 * t_mean * rounding_loss

        # --- onset loss -------------------------------------------------
        if onset_logits is not None and onset_frames is not None:
            onset_loss = onset_bce_loss(
                onset_logits, onset_frames,
                frame_mask=onset_frame_mask,
                pos_weight=self.onset_pos_weight,
            )
        else:
            onset_loss = torch.zeros((), device=self.device)

        return loss, x0_pred, target_ids, count_loss, onset_loss

    @staticmethod
    def _unpack(batch):
        """batch = [token, audio, cond, tab_mask,
                    onset_frames, onset_frame_mask, onset_times, onset_times_mask]
        """
        target, audio, cond, tab_mask = batch[:4]
        extras = list(batch[4:]) + [None] * (4 - len(batch[4:]))
        return (target, audio, cond, tab_mask, *extras)

    def train_step(self, optimizer, optimizer_count_head, optimizer_onset_head, batch):
        optimizer.zero_grad()
        optimizer_count_head.zero_grad()
        optimizer_onset_head.zero_grad()

        (target, audio, cond, tab_mask,
         onset_frames, onset_frame_mask, onset_times, onset_times_mask) = self._unpack(batch)

        loss, x0_pred, target_ids, count_loss, onset_loss = self._forward_pass(
            target, audio, cond, self.sample_timesteps, tab_mask,
            onset_frames, onset_frame_mask, onset_times, onset_times_mask,
        )

        if self.stage == "count":
            count_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.count_head.parameters(), self.max_grad_norm)
            optimizer_count_head.step()
        elif self.stage == "onset":
            onset_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.onset_head.parameters(), self.max_grad_norm)
            optimizer_onset_head.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()

        return loss.item(), count_loss.item(), onset_loss.item()

    def val_step(self, batch):
        (target, audio, cond, tab_mask,
         onset_frames, onset_frame_mask, onset_times, onset_times_mask) = self._unpack(batch)

        loss, x0_pred, target_ids, count_loss, onset_loss = self._forward_pass(
            target, audio, cond, self.sample_timesteps_val, tab_mask,
            onset_frames, onset_frame_mask, onset_times, onset_times_mask,
        )

        acc = self.avg_acc(x0_pred, target_ids, tab_mask)
        return loss.item(), acc, count_loss.item(), onset_loss.item()

    @torch.no_grad()
    def infer_onset_cond(self, audio, cond, max_events, n_events=None):

        spec = cond[..., :-1]
        count_logits = self.count_head(audio, spec)
        onset_logits, _ = self.onset_head(audio, spec, self._count_dist(count_logits))
        if n_events is None:
            n_events = self._n_events(None, count_logits)
        times, valid = decode_onset_times(
            onset_logits, max_events=max_events, n_events=n_events,
            threshold=self.onset_threshold, min_distance=self.onset_min_distance,
        )
        if n_events is None:
            n_events = valid.sum(dim=1)
        return {"prob": torch.sigmoid(onset_logits), "n_events": n_events,
                "times": times, "valid": valid, "n_slots": max_events}

    @torch.no_grad()
    def predict_onsets(self, audio, cond, max_events, n_events=None):
        """Peak-picked onset times only — for metrics and plotting."""
        c = self.infer_onset_cond(audio, cond, max_events, n_events)
        return (None, None) if c is None else (c["times"], c["valid"])


    @torch.no_grad()
    def sample(self, input, audio, cond, tab_mask, num_steps, onset_cond=None,
               event_times=None, event_valid=None, onset_prob=None, n_events=None):
        """Sample new audios from the diffusion model."""
        input = self.encode(input)  # (B, seq_len, embed_dim)
 
        x_noisy = torch.randn_like(input).to(self.device)
        b = x_noisy.shape[0]
        sigmas = LinearSchedule()(num_steps + 1, device=x_noisy.device)
        sigmas_batch = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas_batch, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
 
        sigmas_t_batch = repeat(sigmas, "l -> l b", b=b)
        mask = event_mask_from_tab_mask(tab_mask)  # (B, 1, T) float, 1.0 = real event
 
        # onset conditioning is fixed for the whole trajectory — computed once
        if onset_cond is None and event_times is not None:
            onset_cond = {"times": event_times, "valid": event_valid,
                          "prob": onset_prob, "n_events": n_events,
                          "n_slots": event_times.shape[-1]}
            if n_events is None:
                onset_cond["n_events"] = event_valid.sum(dim=1) if event_valid is not None \
                    else torch.full((b,), float(event_times.shape[-1]), device=self.device)
        if onset_cond is None:
            onset_cond = self.infer_onset_cond(audio, cond, max_events=x_noisy.shape[1])
 
        progress_bar = tqdm(range(num_steps), disable=True)
        # Progressively denoise the audios
        for i in progress_bar:
            v_pred = self.model(x_noisy, sigmas_t_batch[i], audio, cond, mask,
                                onset_cond=onset_cond)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i + 1]:.2f})")
            # if return_process:
            #   intermediate_audios.append(x_noisy.cpu())
        return x_noisy
