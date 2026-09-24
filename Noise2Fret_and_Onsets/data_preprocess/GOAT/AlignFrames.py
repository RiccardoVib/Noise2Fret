"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import os
import glob
import csv
import numpy as np
import librosa
import soundfile as sf


TECHNIQUE_VOCAB = [
    "hammer", "slide", "bend_half", "bend_full", "vibrato", "palm_mute",
    "harmonic", "tapping", "tremolo_picking", "trill",
]

# ─── Helpers  ────────────────────────────────────────

def _flag(row: dict, name: str) -> int:
    v = row.get(name, "")
    return int(float(v)) if v not in ("", None) else 0


def note_technique_set(row: dict) -> list[str]:
    """TECHNIQUE_VOCAB names active for one *_notes.csv row, in vocab order."""
    bend = _flag(row, "bend")
    active = {
        "bend_half": bend == 1,
        "bend_full": bend >= 2,
    }
    return [t for t in TECHNIQUE_VOCAB
            if (active[t] if t in active else _flag(row, t))]


def load_notes(csv_path: str, skip_ties: bool = True, skip_dead: bool = True) -> list[dict]:

    notes = []
    n_legacy_bend = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_techniques = "hammer_from" in (reader.fieldnames or [])
        for row in reader:
            if row.get("type") == "bend_token" or row.get("token") == "bend":
                n_legacy_bend += 1
                continue          # legacy synthetic bend row, not a note
            if skip_ties and row.get("type", "normal") == "tie":
                continue          # ← skip tie targets; their predecessor is already extended
            if skip_dead and row.get("type", "normal") == "dead":
                continue

            notes.append({
                "onset_s":    float(row["onset_s"]),
                "duration_s": float(row["duration_s"]),
                "midi_pitch": int(row["midi_pitch"]),
                "string":     int(row["string"]),
                "fret":       int(row["fret"]),
                "token":      str(row["token"]),
                "is_tie": row.get("type", "normal") == "tie",
                "techniques": note_technique_set(row),
            })
    if n_legacy_bend or not has_techniques:
        print(f"  [warn] {os.path.basename(csv_path)} is a legacy export "
              f"({n_legacy_bend} bend rows skipped, technique columns "
              f"{'present' if has_techniques else 'MISSING'}) -- re-run "
              "TimeTabExtraction.py, techniques will be wrong/empty")
    return notes


def group_notes_by_onset(notes: list[dict]) -> list[list[dict]]:
    """
    Group NOTES by exact onset_s; returns an ordered list of chord groups.
    """
    if not notes:
        return []
    onset_to_notes: dict[float, list[dict]] = {}
    for note in sorted(notes, key=lambda n: n["onset_s"]):
        onset_to_notes.setdefault(note["onset_s"], []).append(note)
    return list(onset_to_notes.values())


def encode_groups(chord_groups: list[list[dict]], window_start: float) -> dict:
    """
    Turn chord groups into the aligned metadata strings.
    """
    tokens, pitches = [], []
    chord_parts, onset_parts, dur_parts, tech_parts = [], [], [], []

    for group in chord_groups:
        chord_parts.append(" ".join(n["token"] for n in group))
        dur_parts.append(" ".join(f"{n['duration_s']:.6f}" for n in group))
        tech_parts.append(" ".join(technique_label(n) for n in group))
        onset_parts.append(round(group[0]["onset_s"] - window_start, 6))
        for n in group:
            tokens.append(n["token"])
            pitches.append(n["midi_pitch"])

    return {
        "n_active": len(tokens),
        "tokens": " ".join(tokens),
        "midi_pitches": ";".join(map(str, pitches)),
        "chord_onsets": ";".join(map(str, onset_parts)),
        "chords": "|".join(chord_parts),
        "chord_durations": "|".join(dur_parts),
        "chord_techniques": "|".join(tech_parts),
    }


def technique_label(note: dict) -> str:
    """'bend_full+vibrato', or '-' for a plain note.  Never contains spaces or '|'."""
    return "+".join(note.get("techniques") or []) or "-"


def parse_chord_techniques(s: str) -> list[list[list[str]]]:
    """Inverse of the `chord_techniques` encoding: groups -> tokens -> names."""
    if not s:
        return []
    return [[[] if e == "-" else e.split("+") for e in g.split(" ")]
            for g in s.split("|")]


# ─── Tempo / bar grid, from TimeTabExtraction.py's sidecar CSVs ────────────────

def load_measures(item_dir: str) -> list[dict]:
    """`<stem>_measures.csv`, or [] when the extractor has not been re-run."""
    paths = glob.glob(os.path.join(item_dir, "*_measures.csv"))
    if not paths:
        return []
    with open(paths[0], newline="", encoding="utf-8") as f:
        return [
            {"start_s": float(r["start_s"]),
             "duration_s": float(r["duration_s"]),
             "bpm": float(r["bpm"]),
             "measure": int(r["measure"])}
            for r in csv.DictReader(f)
        ]


def load_bar_windows(item_dir: str) -> list[dict]:
    paths = glob.glob(os.path.join(item_dir, "*_windows.csv"))
    if not paths:
        return []
    with open(paths[0], newline="", encoding="utf-8") as f:
        return [
            {"window": int(r["window"]),
             "start_s": float(r["start_s"]),
             "end_s": float(r["end_s"]),
             "duration_s": float(r["duration_s"]),
             "bpm": float(r["bpm"]),
             "bars": int(r["bars"]),
             "too_short": int(r["too_short"]),
             "bpm_varies": int(r["bpm_varies"])}
            for r in csv.DictReader(f)
        ]


def bpm_at_time(measures: list[dict], t: float) -> float | None:
    if not measures:
        return None
    bpm = measures[0]["bpm"]
    for m in measures:
        if m["start_s"] <= t:
            bpm = m["bpm"]
        else:
            break
    return bpm


def slice_audio(audio: np.ndarray, sr: int, start_s: float, n_samples: int) -> np.ndarray | None:
    start_sample = int(round(start_s * sr))
    if start_sample >= len(audio):
        return None
    chunk = audio[start_sample:min(start_sample + n_samples, len(audio))]
    if len(chunk) < n_samples:
        chunk = np.pad(chunk, (0, n_samples - len(chunk)))
    return chunk


# ─── Note-centric frame builder ────────────────────────────────────────────────

def build_note_centric_frames(
    notes: list[dict],
    audio: np.ndarray,
    sr: int,
    frame_duration: float = 1.,
    min_note_duration: float = None,
    measures: list[dict] = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    For each unique onset in `notes`, extract a [frame_duration]-second
    audio clip starting at that onset and find all notes active in the window.

    Returns
    -------
    frames   : float32 ndarray [n_frames, frame_samples]
    meta_rows: list of dicts with frame metadata
    """

    frame_samples = int(round(frame_duration * sr))
    measures = measures or []

    # Collect unique onsets (one frame per chord/onset group)
    unique_onsets = sorted({n["onset_s"] for n in notes})

    frame_list = []
    meta_rows  = []

    for frame_idx, anchor_onset in enumerate(unique_onsets):
        f_on  = anchor_onset
        f_off = anchor_onset + frame_duration

        chunk = slice_audio(audio, sr, f_on, frame_samples)
        if chunk is None:
            continue  # onset beyond audio length — skip
        frame_list.append(chunk)

        # ── Active notes in [f_on, f_off) ───────────────────────────
        TAIL_MARGIN = 0.01  # 10 ms

        active_notes = [
            note for note in notes
            if note["onset_s"] < f_off
               and note["onset_s"] > f_on - TAIL_MARGIN
        ]

        active_notes.sort(key=lambda n: (n["onset_s"], n["string"]))

        chord_groups = group_notes_by_onset(active_notes)

        row = {
            "frame_idx": frame_idx,
            "onset_s":   round(f_on,  6),
            "offset_s":  round(f_off, 6),
            "bpm":       bpm_at_time(measures, f_on),
        }
        row.update(encode_groups(chord_groups, f_on))
        meta_rows.append(row)

    frames = np.stack(frame_list).astype(np.float32) if frame_list else np.empty((0, frame_samples), dtype=np.float32)
    return frames, meta_rows


# ─── Bar-aligned frame builder (for the FretNet / Kim baselines) ───────────────

def build_bar_aligned_frames(
    notes: list[dict],
    audio: np.ndarray,
    sr: int,
    windows: list[dict],
    measures: list[dict] = None,
    include_ringing: bool = True,
    skip_too_short: bool = False,
) -> tuple[np.ndarray, list[dict]]:

    measures = measures or []
    if not windows:
        return np.empty((0, 0), dtype=np.float32), []

    use = [w for w in windows if not (skip_too_short and w["too_short"])]
    if not use:
        return np.empty((0, 0), dtype=np.float32), []

    frame_samples = int(round(max(w["duration_s"] for w in use) * sr))

    frame_list, meta_rows = [], []

    for frame_idx, win in enumerate(use):
        f_on, f_off = win["start_s"], win["end_s"]

        chunk = slice_audio(audio, sr, f_on, frame_samples)
        if chunk is None:
            continue
        frame_list.append(chunk)

        active = [n for n in notes if f_on - 0.01 < n["onset_s"] < f_off]
        active.sort(key=lambda n: (n["onset_s"], n["string"]))
        chord_groups = group_notes_by_onset(active)

        row = {
            "frame_idx": frame_idx,
            "onset_s":   round(f_on, 6),
            "offset_s":  round(f_off, 6),
            "valid_s":   round(win["duration_s"], 6),
            "bpm":       win["bpm"] if win["bpm"] else bpm_at_time(measures, f_on),
            "bars":      win["bars"],
            "bpm_varies": win["bpm_varies"],
        }
        row.update(encode_groups(chord_groups, f_on))

        if include_ringing:
            ringing = [
                n for n in notes
                if n["onset_s"] <= f_on - 0.01
                and n["onset_s"] + n["duration_s"] > f_on
            ]
            ringing.sort(key=lambda n: (n["onset_s"], n["string"]))
            row["ringing_chords"] = "|".join(n["token"] for n in ringing)
            row["ringing_durations"] = "|".join(
                f"{min(n['onset_s'] + n['duration_s'], f_off) - f_on:.6f}" for n in ringing
            )
            row["ringing_techniques"] = "|".join(technique_label(n) for n in ringing)
            row["n_ringing"] = len(ringing)
        else:
            row["ringing_chords"] = ""
            row["ringing_durations"] = ""
            row["ringing_techniques"] = ""
            row["n_ringing"] = 0

        meta_rows.append(row)

    frames = (np.stack(frame_list).astype(np.float32) if frame_list
              else np.empty((0, frame_samples), dtype=np.float32))
    return frames, meta_rows


# ─── Per-item processing ───────────────────────────────────────────────────────

META_FIELDS = [
    "item", "audio_file", "frame_idx", "onset_s", "offset_s",
    "n_active", "tokens", "midi_pitches",  "chord_onsets", "chords",
    "chord_durations", "chord_techniques", "bpm",
]

BAR_META_FIELDS = META_FIELDS + [
    "valid_s", "bars", "bpm_varies", "ringing_chords", "ringing_durations", "ringing_techniques", "n_ringing",
]


def process_item(item_dir: str, frame_duration: float = 1., debug: bool = False,
                 mode: str = "onset"):
    item = os.path.basename(item_dir)

    # 1. WAV
    wav_files = glob.glob(os.path.join(item_dir, item + ".wav"))

    if not wav_files:
        print(f"  [skip] No WAV in {item_dir}"); return None

    measures = load_measures(item_dir)
    windows = load_bar_windows(item_dir) if mode == "bars" else []
    if mode == "bars" and not windows:
        print(f"  [skip] No *_windows.csv in {item_dir} -- run TimeTabExtraction first")
        return None
    if not measures:
        print(f"  [warn] No *_measures.csv in {item_dir}; 'bpm' column will be empty")

    all_meta_rows = []
    for wav_path in wav_files:
        # 2. Notes CSV
        note_csvs = glob.glob(os.path.join(item_dir, "*_notes.csv"))
        if not note_csvs:
            print(f"  [skip] No *_notes.csv in {item_dir}"); return None

        # 3. Load audio
        audio, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != 44100:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=44100)
            sr = 44100

        # 4. Load notes
        notes = load_notes(note_csvs[0])
        if not notes:
            print(f"  [skip] Empty notes CSV in {item_dir}"); return None
        print(f"  {os.path.basename(wav_path)} | {len(audio)/sr:.2f}s | {len(notes)} notes")

        # 5. Build frames
        if mode == "bars":
            frames, meta_rows = build_bar_aligned_frames(
                notes, audio, sr, windows, measures=measures
            )
            suffix, fields = "bar", BAR_META_FIELDS
            print(f"  → {len(meta_rows)} bar-aligned frames "
                  f"({windows[0]['bars']} bars, {frames.shape[1]/sr:.2f}s padded)")
        else:
            frames, meta_rows = build_note_centric_frames(
                notes, audio, sr, frame_duration, measures=measures
            )
            suffix, fields = "note", META_FIELDS
            print(f"  → {len(meta_rows)} note-centric frames (1 per unique onset)")

        for r in meta_rows:
            n_groups = len(r["chords"].split("|")) if r["chords"] else 0
            n_onsets = len(r["chord_onsets"].split(";")) if r["chord_onsets"] else 0
            n_durs = len(r["chord_durations"].split("|")) if r["chord_durations"] else 0
            n_tech = len(r["chord_techniques"].split("|")) if r["chord_techniques"] else 0
            assert n_groups == n_onsets == n_durs == n_tech, (
                f"{item} frame {r['frame_idx']}: chords={n_groups} "
                f"onsets={n_onsets} durations={n_durs} techniques={n_tech}"
            )
            # and token-for-token inside every group
            for g_tok, g_tec in zip(r["chords"].split("|"), r["chord_techniques"].split("|")):
                assert len(g_tok.split(" ")) == len(g_tec.split(" ")), (
                    f"{item} frame {r['frame_idx']}: group '{g_tok}' vs '{g_tec}'")
            assert " bend " not in f" {r['tokens']} ", (
                f"{item} frame {r['frame_idx']}: legacy 'bend' token in chords")

        if debug:
            debug_alignment(audio, sr, meta_rows, notes, item)

        # 6. Save .npy
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        npy_path = os.path.join(item_dir, f"{stem}_{suffix}_frames.npy")
        np.save(npy_path, frames)
        print(f"  → frames {frames.shape} → {npy_path}")

        # 7. Save per-item meta CSV
        for r in meta_rows:
            r["item"] = item
            r["audio_file"] = os.path.basename(wav_path)

        meta_csv = os.path.join(item_dir, f"{stem}_{suffix}_frame_meta.csv")
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(meta_rows)
        print(f"  → meta  → {meta_csv}")

        all_meta_rows.extend(meta_rows)

    return meta_rows


# ─── Dataset entry point ───────────────────────────────────────────────────────

def process_dataset_align(root_dir: str, frame_duration: float = 1.,
                          debug: bool = False, mode: str = "onset"):
    item_dirs = sorted(
        glob.glob(os.path.join(root_dir, "item_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not item_dirs:
        raise FileNotFoundError(f"No item_* folders under {root_dir}")

    all_meta = []
    for item_dir in item_dirs:
        print(f"[{os.path.basename(item_dir)}]")
        rows = process_item(item_dir, frame_duration, debug=debug, mode=mode)
        if rows:
            all_meta.extend(rows)

    suffix = "bar" if mode == "bars" else "note"
    fields = BAR_META_FIELDS if mode == "bars" else META_FIELDS
    merged_csv = os.path.join(root_dir, f"all_{suffix}_frame_meta.csv")
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_meta)
    print(f"✓ Done. {len(all_meta)} total {suffix}-frames → {merged_csv}")

    n_missing_bpm = sum(1 for r in all_meta if not r.get("bpm"))
    if n_missing_bpm:
        print(f"  WARNING: {n_missing_bpm} frames have no bpm -- "
              "re-run TimeTabExtraction.py to write *_measures.csv")