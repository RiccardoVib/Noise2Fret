"""
_tab_metrics.py
──────────────
Evaluation metrics for per-string fret prediction (GOATFrameDataset format).

Encoding convention (mirrors GOATFrameDataset):
  class 0 → muted (no note played)
  class 1 → open string (fret 0)
  class 2+ → fret N (class = fret + 1)

Metrics
───────
note_acc      Fraction of string-slots where GT and pred agree on
              whether the string is active (played vs muted),
              ignoring which fret.

fret_acc      Fraction of string-slots where the class ID is exactly
              equal. Stricter than note_acc – correct fret required.

chord_acc     Fraction of time-slots where ALL 6 strings are predicted
              exactly right simultaneously.

false_neg_rate  Of GT-active string-slots, fraction predicted as muted.
false_pos_rate  Of GT-muted string-slots, fraction predicted as active.

pitch_precision / pitch_recall / pitch_f_measure
              Pitch-level P/R/F: predictions are converted to 44-dim
              binary pitch vectors (MIDI 41-84); string / fret identity
              is ignored — only pitch class matters.
              Computed GLOBALLY (sum TP/FP/FN over all batches, then
              divide) to match the reference Metrics.py behaviour.

tab_precision / tab_recall / tab_f_measure
              Tablature P/R/F: (string, fret) pairs must match exactly;
              muted strings are excluded from both numerator and
              denominator. Also computed globally.

tab_disamb    tab_precision / pitch_precision — measures how well the
              model resolves pitch-to-fret ambiguity onto the right string.
"""

import torch

MUTED_CLASS  = 0
STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]
_OPEN_PITCHES = torch.tensor([40, 45, 50, 55, 59, 64], dtype=torch.long)
_PITCH_MIN   = 41          # lowest possible pitch (fret 1 on E2 = 41)
_PITCH_MAX   = 84
_PITCH_DIM   = _PITCH_MAX - _PITCH_MIN + 1  # 44
N_CLASSES = 23

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_ids(x: torch.Tensor) -> torch.Tensor:
    """Accept one-hot (*, 6, C) float or integer IDs (*, 6); return int IDs."""
    if x.ndim == 3 and x.shape[-1] > 6:
        return x.argmax(dim=-1)
    return x.long()


def _ids_to_pitch_vec(ids: torch.Tensor) -> torch.Tensor:
    """
    ids : (..., 6) integer class IDs
    out : (..., 44) binary pitch presence vector (MIDI 41-84)
    """
    if len(ids.shape) == 3:
        ids = torch.argmax(ids, dim=-1)

    ids    = ids.long()
    active = (ids != MUTED_CLASS)                          # (..., 6)
    op     = _OPEN_PITCHES.to(ids.device)                  # (6,)
    midi   = (ids - 1) + op                                # (..., 6) garbage where muted

    *leading, S = ids.shape
    flat_midi   = midi.reshape(-1, S)
    flat_active = active.reshape(-1, S)
    N           = flat_midi.shape[0]

    pitch_vec = torch.zeros(N, _PITCH_DIM, dtype=torch.float32, device=ids.device)
    for s in range(S):
        mask = flat_active[:, s]
        idx  = (flat_midi[:, s] - _PITCH_MIN).clamp(0, _PITCH_DIM - 1)
        pitch_vec[torch.arange(N, device=ids.device)[mask], idx[mask]] = 1.0

    return pitch_vec.reshape(*leading, _PITCH_DIM)          # (..., 44)


def _ids_to_tab_bin(ids: torch.Tensor) -> torch.Tensor:
    """
    ids : (..., 6) integer class IDs
    out : (..., 6, 23) binary (string, fret) presence — muted excluded.
    """
    if len(ids.shape) == 3:
        ids = torch.argmax(ids, dim=-1)

    ids    = ids.long()
    active = (ids != MUTED_CLASS)
    fret   = (ids - 1).clamp(0)

    *leading, S = ids.shape
    flat_fret   = fret.reshape(-1, S)
    flat_active = active.reshape(-1, S)
    N           = flat_fret.shape[0]

    tab_bin = torch.zeros(N, S, 23, dtype=torch.float32, device=ids.device)
    for s in range(S):
        mask = flat_active[:, s]
        idx  = flat_fret[:, s].clamp(0, N_CLASSES-1)
        tab_bin[torch.arange(N, device=ids.device)[mask], s, idx[mask]] = 1.0

    return tab_bin.reshape(*leading, S, N_CLASSES)                 # (..., 6, N_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def tab_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict:
    """
    Compute all tablature metrics for one batch / full dataset.

    Parameters
    ----------
    gt   : (B, T, 6) int | (T, 6) int | (B, T, 6, C) float one-hot
    pred : same shape / type as gt

    Returns
    -------
    dict with scalar keys:
        note_acc, fret_acc, chord_acc,
        false_neg_rate, false_pos_rate,
        pitch_precision, pitch_recall, pitch_f_measure,
        tab_precision, tab_recall, tab_f_measure,
        tab_disamb
    raw TP/denominator keys (used by accumulate_metrics for correct
    global P/R/F aggregation — do NOT average these directly):
        _tp_pitch, _pp_pitch, _pg_pitch,   (pitch TP, pred-pos, gt-pos)
        _tp_tab,   _pp_tab,   _pg_tab,     (tab   TP, pred-pos, gt-pos)
    and per-string (6,) tensor keys:
        note_acc_per_string, fret_acc_per_string,
        false_neg_per_string, false_pos_per_string
    """
    gt_ids   = _to_ids(gt)
    pred_ids = _to_ids(pred)

    if gt.ndim == 3:
        B, T, S = gt.shape
        gt_ids = gt_ids.reshape(B * T, S)
        pred_ids = pred_ids.reshape(B * T, S)


    gt_active   = (gt_ids   != MUTED_CLASS)
    pred_active = (pred_ids != MUTED_CLASS)

    assert gt_ids.shape[-1] == 6

    # ── chord precision / recall / F ──────────────────────────────────────────
    gt_flat = gt_ids.reshape(-1, 6)
    pred_flat = pred_ids.reshape(-1, 6)

    exact_match = (gt_flat == pred_flat).all(dim=-1)  # (N,)
    gt_any_active = (gt_flat != MUTED_CLASS).any(dim=-1)  # GT has ≥1 note
    pred_any_active = (pred_flat != MUTED_CLASS).any(dim=-1)  # pred has ≥1 note

    tp_chord = (exact_match & pred_any_active).float().sum()
    fp_chord = (~exact_match & pred_any_active).float().sum()
    fn_chord = (~exact_match & gt_any_active).float().sum()

    chord_prec = (tp_chord / (tp_chord + fp_chord).clamp(min=1e-12)).item()
    chord_rec = (tp_chord / (tp_chord + fn_chord).clamp(min=1e-12)).item()
    denom_cf = chord_prec + chord_rec
    chord_f = (2 * chord_prec * chord_rec / denom_cf) if denom_cf > 0 else 0.0

    # ── false negatives ───────────────────────────────────────────────────────
    fn_mask          = gt_active & ~pred_active
    false_neg_rate   = (fn_mask.float().sum() /
                        gt_active.float().sum().clamp(min=1)).item()
    false_neg_per_string = (fn_mask.float().reshape(-1, 6).sum(0) /
                            gt_active.float().reshape(-1, 6).sum(0).clamp(min=1))

    # ── false positives ───────────────────────────────────────────────────────
    fp_mask          = ~gt_active & pred_active
    false_pos_rate   = (fp_mask.float().sum() /
                        (~gt_active).float().sum().clamp(min=1)).item()
    false_pos_per_string = (fp_mask.float().reshape(-1, 6).sum(0) /
                            (~gt_active).float().reshape(-1, 6).sum(0).clamp(min=1))

    # ── pitch P / R / F  (global counts) ──────────────────────────────────────
    pv_gt   = _ids_to_pitch_vec(gt_ids)    # (..., 44)
    pv_pred = _ids_to_pitch_vec(pred_ids)  # (..., 44)

    tp_pitch = (pv_gt * pv_pred).sum().item()
    pp_pitch = pv_pred.sum().item()         # predicted positives
    pg_pitch = pv_gt.sum().item()           # ground-truth positives

    pitch_prec = tp_pitch / max(pp_pitch, 1e-12)
    pitch_rec  = tp_pitch / max(pg_pitch, 1e-12)
    denom_pf   = pitch_prec + pitch_rec
    pitch_f    = (2 * pitch_prec * pitch_rec / denom_pf) if denom_pf > 0 else 0.0

    # ── tab P / R / F  (global counts) ────────────────────────────────────────
    tb_gt   = _ids_to_tab_bin(gt_ids)      # (..., 6, N_CLASSES)
    tb_pred = _ids_to_tab_bin(pred_ids)    # (..., 6, N_CLASSES)

    tp_tab = (tb_gt * tb_pred).sum().item()
    pp_tab = tb_pred.sum().item()
    pg_tab = tb_gt.sum().item()

    tab_prec = tp_tab / max(pp_tab, 1e-12)
    tab_rec  = tp_tab / max(pg_tab, 1e-12)
    denom_tf = tab_prec + tab_rec
    tab_f    = (2 * tab_prec * tab_rec / denom_tf) if denom_tf > 0 else 0.0

    # ── tab disambiguation ────────────────────────────────────────────────────
    tab_disamb = (tab_prec / pitch_prec) if pitch_prec > 0 else 0.0

    return {
        # accuracy / rate metrics

        # pitch P/R/F (computed from per-batch counts)
        "pitch_precision":       pitch_prec,
        "pitch_recall":          pitch_rec,
        "pitch_f_measure":       pitch_f,
        # tab P/R/F
        "tab_precision":         tab_prec,
        "tab_recall":            tab_rec,
        "tab_f_measure":         tab_f,
        "tab_disamb":            tab_disamb,
        "chord_prec": chord_prec,
        "chord_rec": chord_rec,
        "chord_f": chord_f,
        "false_neg_rate": false_neg_rate,
        "false_pos_rate": false_pos_rate,
        "false_neg_per_string": false_neg_per_string,
        "false_pos_per_string": false_pos_per_string,
    }




def print_tab_metrics(metrics: dict, save_path: str = None, prefix: str = "") -> None:
    """Pretty-print a metrics dict. If save_path is given, also writes to file."""
    p = f"[{prefix}] " if prefix else ""
    lines = [
        f"{p}{'─' * 62}",
        f"{p}  Pitch precision                    : {metrics['pitch_precision']:.4f}",
        f"{p}  Pitch recall                       : {metrics['pitch_recall']:.4f}",
        f"{p}  Pitch F-measure                    : {metrics['pitch_f_measure']:.4f}",
        f"{p}{'─' * 62}",
        f"{p}  Tab precision                      : {metrics['tab_precision']:.4f}",
        f"{p}  Tab recall                         : {metrics['tab_recall']:.4f}",
        f"{p}  Tab F-measure                      : {metrics['tab_f_measure']:.4f}",
        f"{p}  Tab disambiguation                 : {metrics['tab_disamb']:.4f}",
        f"{p}{'─' * 62}",
        f"{p}  Chord precision     : {metrics['chord_prec']:.4f}",
        f"{p}  Chord recall        : {metrics['chord_rec']:.4f}",
        f"{p}  Chord F-measure      : {metrics['chord_f']:.4f}",
        f"{p}  False-negative  (miss rate)         : {metrics['false_neg_rate']:.4f}",
        f"{p}  False-positive  (ghost-note rate)   : {metrics['false_pos_rate']:.4f}",
        f"{p}{'─' * 62}",
    ]

    text = "\n".join(lines)
    print(text)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)