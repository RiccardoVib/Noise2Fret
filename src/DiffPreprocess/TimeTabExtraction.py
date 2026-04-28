"""
Extract note onsets (in seconds) from GOAT dataset .gp files.

Dataset layout:
    <root>/item_0/
    <root>/item_1/
    ...
    Each folder contains exactly one .gp (or .gp5) file.

Output: one CSV per item  →  <item_folder>/<stem>_notes.csv
  columns: track, string, fret, onset_s, duration_s, midi_pitch, velocity,
           hammer, slide, bend, let_ring, palm_mute, vibrato
And a single merged file  →  <root>/all_notes.csv  (with an extra 'item' column)
"""

import os
import glob
import csv
import guitarpro
from guitarpro import models as gp

QUARTER_TIME = gp.Duration.quarterTime  # 960 ticks per quarter note

# ─── DadaGP instrument group map (MIDI program number → group) ────────────────
# Source: github.com/dada-bots/dadaGP/blob/main/dadagp.py
INSTRUMENT_GROUPS = {
    **{i: "leads"     for i in list(range(0,  24)) + list(range(40, 48)) +
                               list(range(56, 88)) + list(range(96, 100)) +
                               [100] + list(range(108, 120))},
    **{i: "clean"     for i in list(range(24, 29)) + list(range(104, 108))},
    **{i: "distorted" for i in range(29, 32)},
    **{i: "bass"      for i in range(32, 40)},
    **{i: "pads"      for i in list(range(48, 56)) + list(range(88, 96)) + [97, 100, 101, 102, 103]},
    **{i: "remove"    for i in range(120, 128)},
    255: "drums",
}

def get_instrument_group(track) -> str:
    if track.isPercussionTrack:
        return "drums"
    midi_num = track.channel.instrument
    return INSTRUMENT_GROUPS.get(midi_num, "leads")


def build_tracks_by_group(tracks) -> dict:
    groups = {g: [] for g in ["drums", "distorted", "clean", "bass", "leads", "pads", "remove"]}
    for track in tracks:
        groups[get_instrument_group(track)].append(track)
    return groups


def get_instrument_token_prefix(track, tracks_by_group: dict) -> str:
    """Replicate DadaGP's get_instrument_token_prefix exactly."""
    if track in tracks_by_group["drums"]:
        return "drums"
    elif track in tracks_by_group["bass"]:
        return "bass"
    elif track in tracks_by_group["leads"]:
        return "leads"
    elif track in tracks_by_group["pads"]:
        return "pads"
    elif track in tracks_by_group["remove"]:
        return "remove"
    elif track in tracks_by_group["distorted"]:
        for i, t in enumerate(tracks_by_group["distorted"]):
            if track == t:
                return f"distorted{i}"
    elif track in tracks_by_group["clean"]:
        for i, t in enumerate(tracks_by_group["clean"]):
            if track == t:
                return f"clean{i}"
    return "unknown"

def string_value_to_midi(open_tuning_midi: int, fret: int) -> int:
    """Convert open-string MIDI + fret offset to absolute MIDI pitch."""
    return open_tuning_midi + fret

def deduplicate_bend_tokens(notes: list[dict]) -> list[dict]:
    """
    1. Remove a note+bend pair when the note is an exact duplicate (same track,
       string, fret) of the previous note on that string and the previous event
       on that string was already a bend_token  →  muted ghost note after a bend.
    2. Remove any remaining consecutive bend_token on the same (track, string).
    """
    # last event per (track, string): store {"type", "fret"}
    last: dict[tuple, dict] = {}
    result = []
    prev_string = ""
    i = 0
    while i < len(notes):
        note = notes[i]
        key = (note["track"], note["string"])
        prev = last.get(key)
        if note["type"] == "bend_token":
            # Rule 2: skip consecutive bend on same string
            if prev and prev["type"] == "bend_token":
                i += 1
                continue
            # accept this bend, update last
            last[key] = {"type": "bend_token", "fret": note["fret"]}
            result.append(note)

        else:
            # Rule 1: note after a bend with same fret → ghost/muted duplicate
            # also consume the following bend_token if present
            if (
                    prev
                    and prev["type"] == "bend_token"
                    and prev["fret"] == note["fret"]
                    and note["type"] != "bend_token"
                    and note["string"] == prev_string
            ):
                # skip this note; also skip its companion bend_token if next
                i += 1
                if i < len(notes):
                    nxt = notes[i]
                    if (
                            nxt["type"] == "bend_token"
                            and nxt["track"] == note["track"]
                            and nxt["string"] == note["string"]
                    ):
                        i += 1  # skip the companion bend too
                continue  # do not update last[key], bend is still "active"

            last[key] = {"type": note["type"], "fret": note["fret"]}
            result.append(note)

        i += 1
        prev_string = note["string"]
    return result

def merge_tied_notes(notes: list[dict]) -> list[dict]:
    """
    Post-process raw note list:
    - For each tied note (type == 'tie'), find the most recent preceding note
      on the same (track, string) and extend its duration to cover the tie.
    - Tied notes are then removed from the list (no new audio onset).
    """
    # Index non-tied notes by (track, string) for fast lookup
    # We iterate in chronological order; keep a "last seen" pointer per (track, string)
    last_note: dict[tuple, dict] = {}   # (track, string) -> note dict (by reference)
    result = []

    for note in notes:   # already sorted by onset_s
        key = (note["track"], note["string"])
        if note["type"] == "tie":
            prev = last_note.get(key)
            if prev is not None:
                # Extend predecessor's duration to reach the end of this tied beat
                tied_end = note["onset_s"] + note["duration_s"]
                prev_end = prev["onset_s"] + prev["duration_s"]
                prev["duration_s"] = round(max(prev_end, tied_end) - prev["onset_s"], 6)
               # Do NOT append tied note to result — it has no new audio onset
                # If the tied note carries a bend, emit a standalone bend token
                # at the tie's onset (mid-sustain bend — no new pluck)
                if note["bend"] == 1:
                    result.append({**note, "type": "bend_token", "token": "bend"})

        else:
            last_note[key] = note
            result.append(note)

    return result

def extract_notes_from_gp(gp_path: str) -> list[dict]:
    """
    Parse a .gp/.gp5 file and return a list of note dicts with exact
    onset/duration in seconds.

    Strategy for tempo:
      - song.tempo is the initial BPM.
      - MixTableChange events inside beats can update the BPM mid-song.
      - We accumulate wall-clock time by iterating measure headers in order,
        tracking elapsed time at the start of each measure.
      - Inside each measure we walk beats in tick order, converting delta
        ticks to seconds using the *current* BPM.
    """
    song = guitarpro.parse(gp_path)

    # ── Build a tempo map: list of (start_tick, bpm) ──────────────────────
    # start_tick for measure i  =  sum of lengths of measures 0..i-1  + QUARTER_TIME
    # (GP starts the first measure at tick = QUARTER_TIME = 960)
    tempo_map: list[tuple[int, float]] = [(QUARTER_TIME, float(song.tempo))]

    # We need to walk all beats across all tracks to find MixTableChange tempo events.
    # Since tempo is global, one pass over track 0 suffices.
    if song.tracks:
        for measure in song.tracks[0].measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    mtc = beat.effect.mixTableChange if beat.effect else None
                    if mtc and mtc.tempo is not None:
                        tempo_map.append((beat.start, float(mtc.tempo.value)))

    tempo_map.sort(key=lambda x: x[0])

    def ticks_to_seconds(tick: int) -> float:
        """Convert absolute GP tick to seconds using the tempo map."""
        elapsed = 0.0
        prev_tick, prev_bpm = tempo_map[0]
        for seg_tick, seg_bpm in tempo_map[1:]:
            if tick <= seg_tick:
                break
            elapsed += (seg_tick - prev_tick) / QUARTER_TIME * (60.0 / prev_bpm)
            prev_tick, prev_bpm = seg_tick, seg_bpm
        elapsed += (tick - prev_tick) / QUARTER_TIME * (60.0 / prev_bpm)
        return elapsed

    # Build instrument prefix map for all tracks
    tracks_by_group = build_tracks_by_group(song.tracks)

    notes = []
    for track in song.tracks:
        # Build open-string MIDI pitches (GuitarString.value is MIDI pitch of open string)
        open_pitches = {s.number: s.value for s in track.strings}
        prefix = get_instrument_token_prefix(track, tracks_by_group)

        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    if beat.status == gp.BeatStatus.empty:
                        continue
                    onset_s = ticks_to_seconds(beat.start)
                    dur_s   = ticks_to_seconds(beat.start + beat.duration.time) - onset_s

                    for note in beat.notes:
                        if note.type == gp.NoteType.rest:
                            continue

                        open_midi = open_pitches.get(note.string, 40)
                        #midi_pitch = string_value_to_midi(open_midi, note.value)

                        # ── Dead/muted string (X): fret value is meaningless (always 3 in GP) ──
                        is_dead = (note.type == gp.NoteType.dead)
                        fret = -1 if is_dead else note.value  # -1 = dead/muted marker
                        midi_pitch = -1 if is_dead else string_value_to_midi(open_midi, fret)

                        fx = note.effect
                        # ── Grace note attached to this note ──────────────────────────
                        if fx and fx.grace is not None:
                            grace = fx.grace
                            grace_dur_s = grace.durationTime / QUARTER_TIME * (60.0 / song.tempo)
                            grace_onset_s = max(0.0, round(onset_s - grace_dur_s, 6))
                            grace_midi = string_value_to_midi(open_midi, grace.fret)
                            notes.append({
                                "item": os.path.basename(os.path.dirname(gp_path)),
                                "track": track.number,
                                "track_name": track.name.strip(),
                                "string": note.string,  # same string as parent note
                                "fret": grace.fret,
                                "midi_pitch": grace_midi,
                                "onset_s": grace_onset_s,
                                "duration_s": round(grace_dur_s, 6),
                                "velocity": grace.velocity,
                                "type": "grace",
                                "hammer": 0, "slide": 0, "bend": 0,
                                "let_ring": 0, "palm_mute": 0, "vibrato": 0,
                                "token": f"{prefix}:note:s{note.string}:f{grace.fret}",
                            })

                        notes.append({
                            "item":       os.path.basename(os.path.dirname(gp_path)),
                            "track":      track.number,
                            "track_name": track.name.strip(),
                            "string":     note.string,
                            "fret":       fret,
                            "midi_pitch": midi_pitch,
                            "onset_s":    round(onset_s, 6),
                            "duration_s": round(dur_s, 6),
                            "velocity":   note.velocity,
                            "type":       note.type.name,
                            "hammer":     int(fx.hammer)       if fx else 0,
                            "slide":      int(bool(fx.slides)) if fx else 0,
                            "bend":       int(fx.bend is not None) if fx else 0,
                            "let_ring":   int(fx.letRing)      if fx else 0,
                            "palm_mute":  int(fx.palmMute)     if fx else 0,
                            "vibrato":    int(fx.vibrato)      if fx else 0,
                            "token": f"{prefix}:note:s{note.string}:f{fret}",
                        })
                        # ── Bend companion token ───────────────────────────────────────────
                        if fx and fx.bend is not None:
                            notes.append({
                                "item": os.path.basename(os.path.dirname(gp_path)),
                                "track": track.number,
                                "track_name": track.name.strip(),
                                "string": note.string,  # same string → same chord group
                                "fret": note.value,
                                "midi_pitch": midi_pitch,
                                "onset_s": round(onset_s, 6),  # same onset → grouped as simultaneous
                                "duration_s": round(dur_s, 6),
                                "velocity": note.velocity,
                                "type": "bend_token",  # mark it as synthetic
                                "hammer": 0, "slide": 0, "bend": 1,
                                "let_ring": 0, "palm_mute": 0, "vibrato": 0,
                                "token": "bend",
                            })

    notes.sort(key=lambda n: (n["onset_s"], n["track"], n["string"]))
    notes = merge_tied_notes(notes)
    notes = deduplicate_bend_tokens(notes)
    return notes


FIELDNAMES = [
    "item", "track", "track_name", "string", "fret", "midi_pitch",
    "onset_s", "duration_s", "velocity", "type",
    "hammer", "slide", "bend", "let_ring", "palm_mute", "vibrato",
    "token"
]


def process_dataset_extraction(root_dir: str):
    """Walk every item_X folder, extract notes, write per-item CSV + merged CSV."""
    item_dirs = sorted(
        glob.glob(os.path.join(root_dir, "item_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not item_dirs:
        raise FileNotFoundError(f"No item_* folders found under {root_dir}")

    all_notes: list[dict] = []

    for item_dir in item_dirs:
        # Find .gp file (prefer .gp over .gp5 for precision as per task)
        gp_files = glob.glob(os.path.join(item_dir, "*.gp5"))

        if not gp_files:
            print(f"  [skip] No .gp/.gp5 file in {item_dir}")
            continue

        gp_path = gp_files[0]   # take the first match
        item_name = os.path.basename(item_dir)
        print(f"Processing {item_name}: {os.path.basename(gp_path)}")

        try:
            notes = extract_notes_from_gp(gp_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        # ── Per-item CSV ──────────────────────────────────────────────────
        stem = os.path.splitext(os.path.basename(gp_path))[0]
        out_csv = os.path.join(item_dir, f"{stem}_notes.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(notes)
        print(f"  → {len(notes)} notes  →  {out_csv}")

        all_notes.extend(notes)

    # ── Global merged CSV ─────────────────────────────────────────────────
    merged_csv = os.path.join(root_dir, "all_notes.csv")
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_notes)
    print(f"\nDone. {len(all_notes)} total notes → {merged_csv}")


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Extract note onsets from GOAT .gp files")
    # parser.add_argument("root_dir", help="Root directory of the GOAT dataset (contains item_X folders)")
    # args = parser.parse_args()
    # process_dataset(args.root_dir)
    from pathlib import Path
    from Code.Utils.utils import find_folder_upward

    current_dir = Path(os.getcwd())
    print(f"current_dir: {current_dir}")
    files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)

    ROOT_DIR = files_dir / "GOAT_orig/GOAT/"
    process_dataset_extraction(ROOT_DIR)
    ROOT_DIR = files_dir / "GOAT_orig/train/"
    process_dataset_extraction(ROOT_DIR)
    ROOT_DIR = files_dir / "GOAT_orig/test/"
    process_dataset_extraction(ROOT_DIR)
    ROOT_DIR = files_dir / "GOAT_orig/Validation/"
    process_dataset_extraction(ROOT_DIR)

