import torch
import torch.nn.functional as F

OPEN_PITCHES = [40, 45, 50, 55, 59, 64]

def pc_probs_to_soft_binary_fast(tab_logits, open_pitches=OPEN_PITCHES):
    probs = torch.softmax(tab_logits, dim=-1)   # (B, 6, 21)
    B = probs.size(0)

    # Precompute pitch class for each (string, class) pair — shape (6, 21)
    pc_map = torch.zeros(6, 21, dtype=torch.long)
    for s in range(6):
        for cls in range(1, 21):
            pc_map[s, cls] = (open_pitches[s] + cls - 1) % 12
    pc_map = pc_map.to(tab_logits.device)       # (6, 21)

    # Zero out muted class contribution
    mute_mask = torch.zeros(6, 21, device=probs.device)
    mute_mask[:, 1:] = 1.0                      # classes 1–20 active
    probs = probs * mute_mask                   # (B, 6, 21)

    # Flatten strings and classes, scatter-add into 12 pitch classes
    probs_flat  = probs.view(B, -1)             # (B, 126)
    pc_idx_flat = pc_map.view(-1).unsqueeze(0).expand(B, -1)  # (B, 126)

    pc_vec = torch.zeros(B, 12, device=probs.device, dtype=probs.dtype)
    pc_vec.scatter_add_(1, pc_idx_flat, probs_flat)

    return pc_vec.clamp(0.0, 1.0)              # (B, 12)

def jaccard_tonal_distance(pc_a, pc_b):
    """
    pc_a, pc_b: (N, 12) binary pitch class vectors
    Returns: (N,) Jaccard distance in [0, 1]
    """
    intersection = (pc_a * pc_b).sum(-1)
    union = ((pc_a + pc_b).clamp(0, 1)).sum(-1)
    return 1.0 - intersection / (union + 1e-8)

def tonal_loss(encoded_flat, pc_tokens):
    """
    encoded_flat: (B, D)
    pc_tokens: (B, nseq, 6)  values 0–11 or -1 for rests/pads
    """
    B = encoded_flat.shape[0]
    if pc_tokens.dim() == 2:
        pc_tokens = pc_tokens.unsqueeze(1)
    # Build binary pitch class vectors (B, 12)
    mask = (pc_tokens >= 0)                          # (B, nseq, 6)
    pc_vec = torch.zeros(B, 12, device=encoded_flat.device)
    for b in range(B):
        valid_pcs = pc_tokens[b][mask[b]]            # flat list of valid pitch classes
        if valid_pcs.numel() > 0:
            pc_vec[b].scatter_(0, valid_pcs % 12, 1.0)  # binary: 1 if pc present

    # Pairwise Jaccard distance matrix (B, B)
    pc_a = pc_vec.unsqueeze(1).expand(B, B, 12)     # (B, B, 12)
    pc_b = pc_vec.unsqueeze(0).expand(B, B, 12)     # (B, B, 12)
    tonal_dist = jaccard_tonal_distance(
        pc_a.reshape(B * B, 12),
        pc_b.reshape(B * B, 12)
    ).reshape(B, B)                                  # (B, B)

    # Pairwise embedding distance matrix (B, B)
    emb_dist = torch.cdist(encoded_flat.unsqueeze(0),
                           encoded_flat.unsqueeze(0)).squeeze(0)

    # Normalize both to [0, 1] and align
    tonal_dist_norm = tonal_dist / (tonal_dist.max() + 1e-8)
    emb_dist_norm   = emb_dist   / (emb_dist.max()   + 1e-8)

    return F.mse_loss(emb_dist_norm, tonal_dist_norm)


def fret_distance(fret_a, fret_b):
    """
    fret_a, fret_b: (B, 6) — fret positions for one timestep
    Returns: (B,) mean absolute fret distance (ignoring pads)
    """
    dist = (fret_a.float() - fret_b.float()).abs()
    valid = ((fret_a != 0) & (fret_b != 0)).float()
    return dist.sum(-1) / (valid.sum(-1) + 1e-8) # (B,)

def positional_loss(encoded, fret_tokens, pad_fret=-1):
    """
    encoded: (B, D)
    fret_tokens: (B, nseq, 6)
    """
    B = encoded.shape[0]
    # fret_tokens is (B, 6) — no nseq dim, handle directly
    if fret_tokens.dim() == 2:
        fret_tokens = fret_tokens.unsqueeze(1)

    # Use mean fret position across sequence
    mask = ((fret_tokens != pad_fret) & (fret_tokens != 0)).float()  # exclude pads AND open strings
    fret_mean = (fret_tokens.float()*mask).mean(dim=1)  # (B, 6)

    # Pairwise fret distance (B, B) using fret_distance
    fret_a = fret_mean.unsqueeze(1).expand(B, B, 6).reshape(B * B, 6)
    fret_b = fret_mean.unsqueeze(0).expand(B, B, 6).reshape(B * B, 6)

    pos_dist = fret_distance(fret_a, fret_b).reshape(B, B)  # (B, B)

    # Pairwise embedding distance
    emb_dist = torch.cdist(encoded.unsqueeze(0),
                           encoded.unsqueeze(0)).squeeze(0)

    pos_dist_norm = pos_dist / (pos_dist.max() + 1e-8)
    emb_dist_norm = emb_dist / (emb_dist.max() + 1e-8)

    return F.mse_loss(emb_dist_norm, pos_dist_norm)


# Circle of fifths positions for pitch classes 0–11 (C, C#, D, ... B)
COF_POSITIONS = torch.tensor([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=torch.float)
# Index:                       C  C# D  D# E  F   F# G  G# A  A#  B


def cof_chord_distance(pc_vec_a, pc_vec_b,
                       fret_a=None, fret_b=None):
    """
    Circle-of-fifths distance anchored on the bass note (lowest sounding pitch).

    If fret_a / fret_b are provided (B, 6), the lowest note's pitch class is used.
    Falls back to weighted-mean CoF position if no frets are given.

    pc_vec_a, pc_vec_b : (N, 12) binary pitch class vectors
    fret_a, fret_b     : (N, 6)  fret positions, -1 = muted
    Returns            : (N,) distance in [0, 1]
    """
    device = pc_vec_a.device
    cof = COF_POSITIONS.to(device)  # (12,)

    if fret_a is not None and fret_b is not None:
        open_t = torch.tensor(OPEN_PITCHES, device=device, dtype=torch.float)  # (6,)

        def bass_cof(frets):
            # frets: (N, T, 6)
            N, T, _ = frets.shape

            abs_pitch = open_t + frets.float()  # (N, T, 6)
            abs_pitch = abs_pitch.masked_fill(frets < 0, 999.0)  # mute → sentinel
            bass_idx = abs_pitch.argmin(dim=-1)  # (N, T) — lowest string per timestep

            # Build matching index tensors for all three dims
            n_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, T)  # (N, T)
            t_idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)  # (N, T)

            bass_fret = frets[n_idx, t_idx, bass_idx]  # (N, T)
            bass_open = open_t[bass_idx]  # (N, T)
            bass_pc = ((bass_open + bass_fret.float()) % 12).long()  # (N, T)

            # Aggregate CoF position over the time axis → scalar per sample
            return cof[bass_pc].float().mean(dim=1)  # (N,)

        pos_a = bass_cof(fret_a)
        pos_b = bass_cof(fret_b)

    else:
        # fallback: weighted-mean CoF (original behaviour)
        sum_a = pc_vec_a.sum(-1, keepdim=True) + 1e-8
        sum_b = pc_vec_b.sum(-1, keepdim=True) + 1e-8
        pos_a = (pc_vec_a * cof).sum(-1) / sum_a.squeeze(-1)
        pos_b = (pc_vec_b * cof).sum(-1) / sum_b.squeeze(-1)

    diff = (pos_a - pos_b).abs()
    dist = torch.min(diff, 12.0 - diff)   # circular distance in [0, 6]
    return dist / 6.0                      # normalize to [0, 1]


def soft_fret_expectation(tab_logits: torch.Tensor) -> torch.Tensor:
    """(B, 6, 21) → (B, 6) expected fret value per string, differentiable."""
    probs = torch.softmax(tab_logits, dim=-1)              # (B, 6, 21)
    # class 0 = muted (-1), classes 1–20 = frets 0–19
    fret_vals = torch.arange(21, device=probs.device, dtype=probs.dtype) - 1
    return (probs * fret_vals).sum(dim=-1)                 # (B, 6)

def pc_probs_to_soft_binary(
    tab_logits: torch.Tensor,           # (B, 6, 21)  raw logits from model
    open_pitches: list = OPEN_PITCHES,  # [40,45,50,55,59,64] MIDI or just semitone offsets
) -> torch.Tensor:
    """
    Differentiable equivalent of pc_tokens_to_binary for predicted logits.

    Converts (B, 6, 21) logits into a (B, 12) soft pitch-class activation vector.
    Class 0 = muted (excluded), classes 1–20 → frets 0–19.

    Returns values in [0, 1] — soft "probability that pitch class k is active".
    """
    probs = torch.softmax(tab_logits, dim=-1)          # (B, 6, 21)
    B = probs.size(0)
    pc_vec = torch.zeros(B, 12, device=probs.device, dtype=probs.dtype)

    for s in range(6):
        for cls in range(1, 21):                       # skip class 0 (muted)
            fret = cls - 1
            pc = (open_pitches[s] + fret) % 12
            pc_vec[:, pc] = pc_vec[:, pc] + probs[:, s, cls]

    return pc_vec.clamp(0.0, 1.0)                      # (B, 12)

def pc_tokens_to_binary(pc_tokens: torch.Tensor, n_classes: int = 12) -> torch.Tensor:
    """
    Converts raw per-string pitch class indices to a binary chord vector.

    pc_tokens : (B, 6) or (B, nseq, 6), values 0–11 or -1 for muted/pad
    Returns   : (B, 12) binary pitch class vector — 1 if pc is present
    """
    if pc_tokens.dim() == 2:
        pc_tokens = pc_tokens.unsqueeze(1)          # (B, 1, 6)
    B = pc_tokens.shape[0]
    pc_vec = torch.zeros(B, n_classes, device=pc_tokens.device)
    mask = (pc_tokens >= 0)                          # exclude muted / pad
    for b in range(B):
        valid_pcs = pc_tokens[b][mask[b]]            # flat list of active pcs
        if valid_pcs.numel() > 0:
            pc_vec[b].scatter_(0, valid_pcs.long() % 12, 1.0)
    return pc_vec                                    # (B, 12)


def hand_span_penalty(fret_pred, pad_fret=-1, max_span=6):
    """
    fret_pred: (B, 6) predicted fret values for one timestep
    Returns:   (B,) penalty in [0, 1], 0 if span <= max_span
    """
    mask = (fret_pred != pad_fret) & (fret_pred != 0)  # exclude pads and open strings

    # Replace non-fretted positions with NaN-equivalent for min/max
    frets_masked = fret_pred.float().clone()
    frets_masked[~mask] = float('nan')

    fret_max = torch.where(mask.any(-1), frets_masked.nan_to_num(-999).max(-1).values,
                           torch.zeros(fret_pred.shape[0], device=fret_pred.device))
    fret_min = torch.where(mask.any(-1), frets_masked.nan_to_num(999).min(-1).values,
                           torch.zeros(fret_pred.shape[0], device=fret_pred.device))

    span = (fret_max - fret_min).abs() # (B,)
    excess = (span - max_span).clamp(min=0)  # 0 if within limit, positive if over
    return excess / max_span  # normalize: max excess = ~24/5

def string_activity_jaccard_loss(fret_pred, fret_target, pad_fret=-1):
    """
    Returns scalar Jaccard distance on string activity vectors.
    0 = perfect match, 1 = no overlap at all.
    """
    played_pred   = (fret_pred   != pad_fret).float()   # (B, 6)
    played_target = (fret_target != pad_fret).float()   # (B, 6)

    intersection = (played_pred * played_target).sum(-1)          # (B,)
    union        = (played_pred + played_target).clamp(0,1).sum(-1)  # (B,)

    jaccard = intersection / (union + 1e-8)             # (B,) similarity
    return (1.0 - jaccard).mean()
