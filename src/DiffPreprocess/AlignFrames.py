"""
align_frames_note_centric.py

Note-centric version of align_frames.py.
Instead of sliding a fixed-hop window and finding active notes,
we iterate over the notes CSV, group by onset, and for each unique
onset create a 1-second audio frame anchored at that onset.

For each frame we collect ALL notes whose active window overlaps
[onset_s, onset_s + FRAME_DURATION), preserving chord (simultaneous
note) structure in the 'chords' column.

Output per item:
  - <stem>_note_frames.npy     : float32 [n_frames, frame_samples]
  - <stem>_note_frame_meta.csv : frame_idx, onset_s, offset_s,
                                  n_active, tokens, midi_pitches, chords
Merged:
  - all_note_frame_meta.csv
"""

import os
import glob
import csv
import numpy as np
import librosa
import soundfile as sf

FRAME_DURATION = 1.0  # seconds

# ─── Helpers (unchanged from original) ────────────────────────────────────────

def load_notes(csv_path: str, skip_ties: bool = True, skip_dead: bool = True) -> list[dict]:

    notes = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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
            })
    return notes


def group_notes_by_onset(notes: list[dict]) -> list[list[str]]:
    """Group tokens by exact onset_s; returns ordered list of chord groups."""
    if not notes:
        return []
    onset_to_tokens: dict[float, list[str]] = {}
    for note in sorted(notes, key=lambda n: n["onset_s"]):
        onset_to_tokens.setdefault(note["onset_s"], []).append(note["token"])
    return list(onset_to_tokens.values())


def split_bend_tokens(chord_groups: list[list[str]]) -> list[list[str]]:
    """
    For each chord group, split 'bend' tokens into a separate subsequent group.
    e.g. ["distorted0:note:s2:f7", "bend"] → ["distorted0:note:s2:f7"] | ["bend"]
    """
    result = []
    for group in chord_groups:
        notes = [t for t in group if t != "bend"]
        bends = [t for t in group if t == "bend"]
        if notes:
            result.append(notes)
        if bends:
            result.append(bends)
    return result

# ─── Note-centric frame builder ────────────────────────────────────────────────

def build_note_centric_frames(
    notes: list[dict],
    audio: np.ndarray,
    sr: int,
    frame_duration: float = FRAME_DURATION,
    min_note_duration: float = None,   # fallback minimum note duration (defaults to 1 hop ≈ 1/sr)
) -> tuple[np.ndarray, list[dict]]:
    """
    For each unique onset in `notes`, extract a [frame_duration]-second
    audio clip starting at that onset and find all notes active in the window.

    Returns
    -------
    frames   : float32 ndarray [n_frames, frame_samples]
    meta_rows: list of dicts with frame metadata
    """
    if min_note_duration is None:
        min_note_duration = 1.0 / sr  # at least 1 sample

    frame_samples = int(round(frame_duration * sr))
    n_audio = len(audio)

    # Collect unique onsets (one frame per chord/onset group)
    unique_onsets = sorted({n["onset_s"] for n in notes})

    frame_list = []
    meta_rows  = []

    for frame_idx, anchor_onset in enumerate(unique_onsets):
        f_on  = anchor_onset
        f_off = anchor_onset + frame_duration

        # ── Audio slice ──────────────────────────────────────────────
        start_sample = int(round(f_on * sr))
        end_sample   = start_sample + frame_samples

        if start_sample >= n_audio:
            continue  # onset beyond audio length — skip

        # Pad with zeros if the 1-second window extends past the end
        chunk = audio[start_sample:min(end_sample, n_audio)]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))

        frame_list.append(chunk)

        # ── Active notes in [f_on, f_off) ───────────────────────────
        TAIL_MARGIN = 0.01  # 1 ms

        active_notes = [
            note for note in notes
            if note["onset_s"] <= f_off - TAIL_MARGIN
               and note["onset_s"] + max(note["duration_s"], min_note_duration) > f_on + TAIL_MARGIN
        ]

        active_notes.sort(key=lambda n: (n["onset_s"], n["string"]))
        active_tokens  = [n["token"]      for n in active_notes]
        active_pitches = [n["midi_pitch"] for n in active_notes]

        # Chord groups: group active notes by onset → "tok1 tok2|tok3|..."
        chord_groups = group_notes_by_onset(active_notes)
        chord_groups = [group for group in chord_groups if not all(t == "bend" for t in group)]
        #chord_groups = split_bend_tokens(chord_groups)
        chords_str   = "|".join(" ".join(g) for g in chord_groups)

        meta_rows.append({
            "frame_idx":      frame_idx,
            "onset_s":        round(f_on,  6),
            "offset_s":       round(f_off, 6),
            "n_active":       len(active_notes),
            "tokens":         " ".join(active_tokens),
            "midi_pitches":   ";".join(map(str, active_pitches)),
            "chords":         chords_str,
        })

    frames = np.stack(frame_list).astype(np.float32) if frame_list else np.empty((0, frame_samples), dtype=np.float32)
    return frames, meta_rows


# ─── Per-item processing ───────────────────────────────────────────────────────

META_FIELDS = [
    "item", "frame_idx", "onset_s", "offset_s",
    "n_active", "tokens", "midi_pitches", "chords"
]


def process_item(item_dir: str, frame_duration: float = FRAME_DURATION, debug: bool = False):
    item = os.path.basename(item_dir)

    # 1. WAV
    wav_files = glob.glob(os.path.join(item_dir, item + ".wav"))
    if not wav_files:
        print(f"  [skip] No WAV in {item_dir}"); return None
    wav_path = wav_files[0]

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

    # 5. Build note-centric frames
    frames, meta_rows = build_note_centric_frames(notes, audio, sr, frame_duration)
    n_frames = len(meta_rows)
    print(f"  → {n_frames} note-centric frames (1 per unique onset)")

    if debug:
        debug_alignment(audio, sr, meta_rows, notes, item)

    # 6. Save .npy
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    npy_path = os.path.join(item_dir, f"{stem}_note_frames.npy")
    np.save(npy_path, frames)
    print(f"  → frames {frames.shape} → {npy_path}")

    # 7. Save per-item meta CSV
    for r in meta_rows:
        r["item"] = item
    meta_csv = os.path.join(item_dir, f"{stem}_note_frame_meta.csv")
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"  → meta  → {meta_csv}")

    return meta_rows


# ─── Dataset entry point ───────────────────────────────────────────────────────

def process_dataset_align(root_dir: str, frame_duration: float = FRAME_DURATION, debug: bool = False):
    item_dirs = sorted(
        glob.glob(os.path.join(root_dir, "item_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not item_dirs:
        raise FileNotFoundError(f"No item_* folders under {root_dir}")

    all_meta = []
    for item_dir in item_dirs:
        print(f"[{os.path.basename(item_dir)}]")
        rows = process_item(item_dir, frame_duration, debug=debug)
        if rows:
            all_meta.extend(rows)

    merged_csv = os.path.join(root_dir, "all_note_frame_meta.csv")
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(all_meta)
    print(f"✓ Done. {len(all_meta)} total note-frames → {merged_csv}")


# ─── Debug ─────────────────────────────────────────────────────────────────────

def debug_alignment(audio, sr, meta_rows, notes, item, n_examples=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_dir = os.path.join(os.getcwd(), f"debug_nc_{item}")
    os.makedirs(debug_dir, exist_ok=True)

    sample = meta_rows[::max(1, len(meta_rows) // n_examples)][:n_examples]
    print(f"{'─'*70}")
    print(f"  DEBUG [{item}]: showing {len(sample)} note-centric frames → {debug_dir}")

    fig, axes = plt.subplots(len(sample), 1, figsize=(12, 3 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    for ax, row in zip(axes, sample):
        f_on  = row["onset_s"]
        f_off = row["offset_s"]
        s_start = int(round(f_on * sr))
        frame_audio = audio[s_start : s_start + int(round((f_off - f_on) * sr))]

        print(f"  Frame {row['frame_idx']:>6d} anchor={f_on:.4f}s  n_active={row['n_active']}")
        print(f"    tokens : {row['tokens']}")
        print(f"    chords : {row['chords']}\n")

        t = np.linspace(f_on, f_on + len(frame_audio) / sr, len(frame_audio))
        ax.plot(t, frame_audio, lw=0.6, color="steelblue")
        ax.axvline(f_on, color="green", lw=1.5, linestyle="-", label="anchor onset")
        ax.set_xlim(f_on, f_off)
        ax.set_title(f"Frame {row['frame_idx']} anchor={f_on:.4f}s | {row['tokens']}", fontsize=8)

        for note in notes:
            if f_on <= note["onset_s"] < f_off:
                ax.axvline(note["onset_s"], color="crimson", lw=1.0, linestyle="--",
                           label=note["token"])
        ax.legend(fontsize=7, loc="upper right")

        wav_out = os.path.join(debug_dir, f"frame_{row['frame_idx']:06d}.wav")
        sf.write(wav_out, frame_audio, sr)

    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "alignment_overview.png"), dpi=120)
    plt.close()


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from Code.Utils.utils import find_folder_upward

    current_dir = Path(os.getcwd())
    files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)

    ROOT_DIR = files_dir / "GOAT_orig/GOAT"
    process_dataset_align(ROOT_DIR, frame_duration=FRAME_DURATION, debug=False)

    ROOT_DIR = files_dir / "GOAT_orig/train"
    process_dataset_align(ROOT_DIR, frame_duration=FRAME_DURATION)

    ROOT_DIR = files_dir / "GOAT_orig/test"
    process_dataset_align(ROOT_DIR, frame_duration=FRAME_DURATION, debug=False)
    ROOT_DIR = files_dir / "GOAT_orig/Validation/"
    process_dataset_align(ROOT_DIR, frame_duration=FRAME_DURATION, debug=True)
