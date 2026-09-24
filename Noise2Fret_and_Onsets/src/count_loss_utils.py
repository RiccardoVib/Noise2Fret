"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""
import torch
import torch.nn.functional as F


def build_neighbor_smoothed_targets(
    target: torch.Tensor, num_classes: int, smoothing: float = 0.1
) -> torch.Tensor:
    """
    target : (B,) long, hard class indices in [0, num_classes)
    returns: (B, num_classes) float, soft target distribution.
    """
    device = target.device
    B = target.shape[0]
    dist = torch.zeros(B, num_classes, device=device, dtype=torch.float32)

    center_w = 1.0 - smoothing
    edge_w = smoothing / 2.0

    idx = target.unsqueeze(1)  # (B, 1)
    dist.scatter_(1, idx, center_w)

    left_idx = target - 1
    right_idx = target + 1
    left_valid = (left_idx >= 0)
    right_valid = (right_idx < num_classes)

    li = left_idx.clamp(min=0).unsqueeze(1)
    ri = right_idx.clamp(max=num_classes - 1).unsqueeze(1)

    dist.scatter_add_(1, li, (edge_w * left_valid.float()).unsqueeze(1))
    dist.scatter_add_(1, ri, (edge_w * right_valid.float()).unsqueeze(1))

    # give back any mass that had nowhere to go (boundary classes)
    lost = edge_w * (~left_valid).float() + edge_w * (~right_valid).float()
    dist.scatter_add_(1, idx, lost.unsqueeze(1))

    return dist


def soft_cross_entropy(
    logits: torch.Tensor,
    target_dist: torch.Tensor,
    target_hard: torch.Tensor = None,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    logits       : (B, num_classes)
    target_dist  : (B, num_classes) soft targets (e.g. from
                   build_neighbor_smoothed_targets)
    target_hard  : (B,) long, the original hard class index — required if
                   class_weights is given, since weighting is applied per
                   the *true* class, same convention as
                   F.cross_entropy(weight=...).
    class_weights: (num_classes,) float, e.g. from
                   compute_inverse_freq_class_weights.

    returns scalar loss (weighted mean over the batch, matching
    F.cross_entropy's default reduction='mean' + weight behavior).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(target_dist * log_probs).sum(dim=-1)  # (B,)

    if class_weights is not None:
        assert target_hard is not None, "target_hard is required when using class_weights"
        w = class_weights.to(logits.device)[target_hard]  # (B,)
        return (per_sample * w).sum() / w.sum().clamp(min=1e-8)

    return per_sample.mean()


def compute_inverse_freq_class_weights(
    class_counts: torch.Tensor, scheme: str = "sqrt_inv", eps: float = 1.0, max_ratio: float = 5.0

) -> torch.Tensor:
    """
    class_counts : (num_classes,) — number of training examples per class
                   (can include zeros for classes that never occur).
    scheme       : "inv"      -> weight ∝ 1 / count
                   "sqrt_inv" -> weight ∝ 1 / sqrt(count)   (gentler, usually
                                 preferred — plain inverse-frequency can
                                 massively overweight rare classes and make
                                 training unstable)
    eps          : additive smoothing so zero-count classes don't produce inf.

    returns (num_classes,) float, normalized so the mean weight is 1.0
    (keeps the overall loss scale comparable to unweighted CE).
    """
    counts = class_counts.float() + eps
    if scheme == "inv":
        w = 1.0 / counts
    elif scheme == "sqrt_inv":
        w = 1.0 / counts.sqrt()
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    w = w * (len(w) / w.sum())  # normalize so mean(w) == 1
    if max_ratio is not None:
        w = w.clamp(max=max_ratio)
        w = w * (len(w) / w.sum())  # renormalize after clamping

    return w


@torch.no_grad()
def compute_count_class_counts_from_dataloader(
    dataloader, num_classes: int, zero_events_impossible: bool = True, device="cpu"
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in dataloader:
        tab_mask = batch["tab_mask"]
        n_real = (~tab_mask).view(tab_mask.shape[0], -1).sum(dim=1) // 6
        target = (n_real - 1).long() if zero_events_impossible else n_real.long()
        target = target.clamp(min=0, max=num_classes - 1)
        counts += torch.bincount(target, minlength=num_classes)
    return counts
