"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import os
import glob
import csv
import json
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

BEND_UNITS_PER_SEMITONE = 2
BEND_FULL_ABOVE_SEMITONES = 1.0    # > 1 semitone -> full; quarter bends count as half

TECHNIQUE_COLUMNS = [
    "hammer", "hammer_from", "slide", "bend", "vibrato", "palm_mute",
    "let_ring", "harmonic", "tapping", "trill", "tremolo_picking",
    "staccato", "ghost", "accent",
]

_TIE_INHERITED = ("slide", "vibrato", "palm_mute", "let_ring", "harmonic",
                  "trill", "tremolo_picking", "hammer_from")

_SLIDE_ANY = {gp.SlideType.shiftSlideTo, gp.SlideType.legatoSlideTo,
              gp.SlideType.intoFromAbove, gp.SlideType.intoFromBelow,
              gp.SlideType.outDownwards, gp.SlideType.outUpwards}


def bend_class(bend) -> int:
    """0 none / 1 half / 2 full, from a pyguitarpro BendEffect."""
    if bend is None:
        return 0
    peak_units = max((abs(p.value) for p in bend.points), default=0)
    if peak_units == 0:
        # curve present but flat (malformed / release written as 0 points):
        # fall back to the header amplitude (raw GP units, 50 per semitone)
        peak_semitones = abs(bend.value) / 50.0
    else:
        peak_semitones = peak_units / BEND_UNITS_PER_SEMITONE
    if peak_semitones <= 0:
        return 1          # a bend is notated but carries no amplitude: call it half
    return 2 if peak_semitones > BEND_FULL_ABOVE_SEMITONES else 1


def note_techniques(note, beat) -> dict:
    """Technique flags for one GP note (before legato attribution / tie merge)."""
    fx = note.effect
    bfx = beat.effect
    t = {c: 0 for c in TECHNIQUE_COLUMNS}
    if fx is not None:
        t["hammer_from"] = int(bool(fx.hammer))
        t["slide"] = int(any(s in _SLIDE_ANY for s in (fx.slides or [])))
        t["bend"] = bend_class(fx.bend)
        t["vibrato"] = int(bool(fx.vibrato))
        t["palm_mute"] = int(bool(fx.palmMute))
        t["let_ring"] = int(bool(fx.letRing))
        t["harmonic"] = int(fx.harmonic is not None)
        t["trill"] = int(fx.trill is not None)
        t["tremolo_picking"] = int(fx.tremoloPicking is not None)
        t["staccato"] = int(bool(fx.staccato))
        t["ghost"] = int(bool(fx.ghostNote))
        t["accent"] = int(bool(fx.accentuatedNote or fx.heavyAccentuatedNote))
        # a grace note leading into this note with a transition
        g = fx.grace
        if g is not None:
            if g.transition == gp.GraceEffectTransition.hammer:
                t["hammer"] = 1
            elif g.transition == gp.GraceEffectTransition.slide:
                t["slide"] = 1
            elif g.transition == gp.GraceEffectTransition.bend and t["bend"] == 0:
                t["bend"] = 2 if abs(note.value - g.fret) > 1 else 1
    if bfx is not None:
        t["vibrato"] |= int(bool(bfx.vibrato))
        t["tapping"] = int(bfx.slapEffect == gp.SlapEffect.tapping)
    return t


def attribute_legato_targets(notes: list[dict]) -> list[dict]:
    """
    Move GuitarPro's hammer/pull-off flag from the arc's origin to its target.

    GP writes `hammer` on the note the arc STARTS from; the note heard without a
    pick attack is the NEXT note on the same (track, string).  Ties are skipped
    when looking for the target (a tie continues the origin, it is not a new
    pitch).  `notes` must be sorted chronologically.
    """
    by_string: dict[tuple, list[dict]] = {}
    for n in notes:
        if n["type"] in ("tie", "dead"):
            continue
        by_string.setdefault((n["track"], n["string"]), []).append(n)

    # an origin may itself be a tie (sustained note that then hammers on)
    origins = [n for n in notes if n["hammer_from"]]
    for o in origins:
        seq = by_string.get((o["track"], o["string"]), [])
        nxt = next((n for n in seq if n["onset_s"] > o["onset_s"]), None)
        if nxt is not None:
            nxt["hammer"] = 1
    return notes


def merge_tied_notes(notes: list[dict]) -> list[dict]:
    """
    - For each tied note (type == 'tie'), find the most recent preceding note
      on the same (track, string) and extend its duration to cover the tie.
    - Effects written on the tie are merged INTO that parent note (vibrato,
      slide, ...; bend takes the larger of the two classes).  The tie row is
      then dropped.  Previously a bent tie emitted a bare `bend` row with no
      parent note -- the source of the bend-only rows.
    """
    last_note: dict[tuple, dict] = {}   # (track, string) -> note dict (by reference)
    result = []
    n_orphan = 0

    for note in notes:   # already sorted by onset_s
        key = (note["track"], note["string"])
        if note["type"] == "tie":
            prev = last_note.get(key)
            if prev is None:
                n_orphan += 1          # tie with nothing to tie to: drop it
                continue
            tied_end = note["onset_s"] + note["duration_s"]
            prev_end = prev["onset_s"] + prev["duration_s"]
            prev["duration_s"] = round(max(prev_end, tied_end) - prev["onset_s"], 6)
            for c in _TIE_INHERITED:
                prev[c] = prev[c] | note[c]
            prev["bend"] = max(prev["bend"], note["bend"])
        else:
            if note["type"] != "grace":   # a grace note never carries a tie
                last_note[key] = note
            result.append(note)

    if n_orphan:
        print(f"    [warn] {n_orphan} tie(s) with no preceding note on their string, dropped")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Tempo map and bar grid
# ══════════════════════════════════════════════════════════════════════════════

def build_tempo_map(song) -> list[tuple[int, float]]:
    """
    [(start_tick, bpm), ...] sorted by tick.

    GP starts the first measure at tick = QUARTER_TIME = 960, and tempo can
    change mid-song via MixTableChange.  Tempo is global, so one pass over
    track 0 suffices.
    """
    tempo_map: list[tuple[int, float]] = [(QUARTER_TIME, float(song.tempo))]

    if song.tracks:
        for measure in song.tracks[0].measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    mtc = beat.effect.mixTableChange if beat.effect else None
                    if mtc and mtc.tempo is not None:
                        tempo_map.append((beat.start, float(mtc.tempo.value)))

    tempo_map.sort(key=lambda x: x[0])
    return tempo_map


def make_tick_converter(tempo_map: list[tuple[int, float]]):
    """Returns (ticks_to_seconds, bpm_at_tick)."""

    def ticks_to_seconds(tick: int) -> float:
        elapsed = 0.0
        prev_tick, prev_bpm = tempo_map[0]
        for seg_tick, seg_bpm in tempo_map[1:]:
            if tick <= seg_tick:
                break
            elapsed += (seg_tick - prev_tick) / QUARTER_TIME * (60.0 / prev_bpm)
            prev_tick, prev_bpm = seg_tick, seg_bpm
        elapsed += (tick - prev_tick) / QUARTER_TIME * (60.0 / prev_bpm)
        return elapsed

    def bpm_at_tick(tick: int) -> float:
        bpm = tempo_map[0][1]
        for seg_tick, seg_bpm in tempo_map:
            if seg_tick <= tick:
                bpm = seg_bpm
            else:
                break
        return bpm

    return ticks_to_seconds, bpm_at_tick


def extract_measures(song, ticks_to_seconds, bpm_at_tick) -> list[dict]:
    """
    The bar grid.  One row per measure, in order.
    """
    measures = []
    if not song.tracks:
        return measures

    for measure in song.tracks[0].measures:
        start_tick = measure.start
        length_tick = measure.length
        start_s = ticks_to_seconds(start_tick)
        end_s = ticks_to_seconds(start_tick + length_tick)
        ts = measure.timeSignature
        measures.append({
            "measure": measure.number,
            "start_tick": start_tick,
            "length_tick": length_tick,
            "start_s": round(start_s, 6),
            "duration_s": round(end_s - start_s, 6),
            "bpm": bpm_at_tick(start_tick),
            "numerator": ts.numerator,
            "denominator": ts.denominator.value,
        })
    return measures


def _group_windows(
    measures: list[dict],
    bars_per_window: int,
    min_seconds: float,
    require_constant_bpm: bool,
) -> list[dict]:
    windows = []
    for w, i in enumerate(range(0, len(measures) - bars_per_window + 1, bars_per_window)):
        group = measures[i:i + bars_per_window]
        start_s = group[0]["start_s"]
        end_s = group[-1]["start_s"] + group[-1]["duration_s"]
        bpms = {m["bpm"] for m in group}
        duration = round(end_s - start_s, 6)
        windows.append({
            "window": w,
            "start_s": start_s,
            "end_s": round(end_s, 6),
            "duration_s": duration,
            "bpm": group[0]["bpm"],
            "bars": bars_per_window,
            "first_measure": group[0]["measure"],
            "last_measure": group[-1]["measure"],
            "too_short": int(duration <= min_seconds),
            "bpm_varies": int(require_constant_bpm and len(bpms) > 1),
        })
    return windows


def plan_bar_windows(
    measures: list[dict],
    bars_per_window: int = 4,
    min_seconds: float = 4.65,
    require_constant_bpm: bool = True,
    adaptive: bool = True,
    max_bars: int = 16,
) -> list[dict]:

    best = _group_windows(measures, bars_per_window, min_seconds, require_constant_bpm)
    if not adaptive:
        return best

    bars = bars_per_window
    while bars <= max_bars:
        windows = _group_windows(measures, bars, min_seconds, require_constant_bpm)
        if not windows:
            break                      # too few measures to make even one window
        best = windows
        if all(w["too_short"] == 0 for w in windows):
            return windows
        bars *= 2
    return best


def bpm_at_time(measures: list[dict], t_seconds: float) -> float:

    bpm = measures[0]["bpm"] if measures else 120.0
    for m in measures:
        if m["start_s"] <= t_seconds:
            bpm = m["bpm"]
        else:
            break
    return bpm


# ══════════════════════════════════════════════════════════════════════════════


def extract_notes_from_gp(gp_path: str) -> tuple[list[dict], list[dict], dict]:
    """
    Parse a .gp/.gp5 file.

    Returns (notes, measures, meta).  Previously returned only ``notes``; the
    two extra values carry the bar grid and the tempo map.
    """
    song = guitarpro.parse(gp_path)

    tempo_map = build_tempo_map(song)
    ticks_to_seconds, bpm_at_tick = make_tick_converter(tempo_map)
    measures = extract_measures(song, ticks_to_seconds, bpm_at_tick)

    # measure lookup by tick, for stamping each note with its bar
    measure_starts = [(m["start_tick"], m["measure"], m["length_tick"]) for m in measures]

    def measure_of(tick: int) -> tuple[int, float]:
        """(measure number, position within the bar in quarter notes)."""
        number, start = (measure_starts[0][1], measure_starts[0][0]) if measure_starts else (0, 0)
        for m_start, m_num, _ in measure_starts:
            if m_start <= tick:
                number, start = m_num, m_start
            else:
                break
        return number, round((tick - start) / QUARTER_TIME, 6)

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

                    # tempo and bar position for this beat
                    beat_bpm = bpm_at_tick(beat.start)
                    measure_no, beat_in_measure = measure_of(beat.start)

                    for note in beat.notes:
                        if note.type == gp.NoteType.rest:
                            continue

                        open_midi = open_pitches.get(note.string, 40)

                        # ── Dead/muted string (X): fret value is meaningless (always 3 in GP) ──
                        is_dead = (note.type == gp.NoteType.dead)
                        fret = -1 if is_dead else note.value  # -1 = dead/muted marker
                        midi_pitch = -1 if is_dead else string_value_to_midi(open_midi, fret)

                        fx = note.effect
                        # ── Grace note attached to this note ──────────────────────────
                        if fx and fx.grace is not None:
                            grace = fx.grace
                            grace_dur_s = grace.durationTime / QUARTER_TIME * (60.0 / beat_bpm)
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
                                "bpm": beat_bpm,
                                "measure": measure_no,
                                "beat_in_measure": beat_in_measure,
                                **{c: 0 for c in TECHNIQUE_COLUMNS},
                                "ghost": int(bool(grace.isDead)),
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
                            "bpm":        beat_bpm,
                            "measure":    measure_no,
                            "beat_in_measure": beat_in_measure,
                            **note_techniques(note, beat),
                            "token": f"{prefix}:note:s{note.string}:f{fret}",
                        })


    notes.sort(key=lambda n: (n["onset_s"], n["track"], n["string"]))
    notes = attribute_legato_targets(notes)   # needs the ties still in place
    notes = merge_tied_notes(notes)

    bpms = sorted({m["bpm"] for m in measures})
    time_sigs = sorted({f"{m['numerator']}/{m['denominator']}" for m in measures})
    total_s = (measures[-1]["start_s"] + measures[-1]["duration_s"]) if measures else 0.0

    meta = {
        "item": os.path.basename(os.path.dirname(gp_path)),
        "file": os.path.basename(gp_path),
        "initial_bpm": float(song.tempo),
        "bpm_values": bpms,
        "bpm_constant": len(bpms) <= 1,
        "tempo_map": [
            {"tick": t, "start_s": round(ticks_to_seconds(t), 6), "bpm": b}
            for t, b in tempo_map
        ],
        "time_signatures": time_sigs,
        "n_measures": len(measures),
        "duration_s": round(total_s, 6),
        "n_notes": len(notes),
    }
    return notes, measures, meta


NOTE_FIELDNAMES = [
    "item", "track", "track_name", "string", "fret", "midi_pitch",
    "onset_s", "duration_s", "velocity", "type",
    "bpm", "measure", "beat_in_measure",
    *TECHNIQUE_COLUMNS,
    "token"
]
FIELDNAMES = NOTE_FIELDNAMES

MEASURE_FIELDNAMES = [
    "item", "measure", "start_tick", "length_tick",
    "start_s", "duration_s", "bpm", "numerator", "denominator",
]

WINDOW_FIELDNAMES = [
    "item", "window", "start_s", "end_s", "duration_s", "bpm", "bars",
    "first_measure", "last_measure", "too_short", "bpm_varies",
]


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_technique_stats(notes: list[dict]):
    """Per-technique note counts, to decide which classes the head can learn."""
    played = [n for n in notes if n["type"] != "dead"]
    if not played:
        return
    print(f"\nTechnique frequency over {len(played)} played notes:")
    rows = [(c, sum(1 for n in played if n[c])) for c in TECHNIQUE_COLUMNS if c != "bend"]
    rows += [("bend_half", sum(1 for n in played if n["bend"] == 1)),
             ("bend_full", sum(1 for n in played if n["bend"] == 2))]
    for name, k in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<16s} {k:>8d}  ({100 * k / len(played):5.2f} %)")
    n_plain = sum(1 for n in played
                  if not any(n[c] for c in TECHNIQUE_COLUMNS if c != "hammer_from"))
    print(f"  {'(none)':<16s} {n_plain:>8d}  ({100 * n_plain / len(played):5.2f} %)")


def process_dataset_extraction(root_dir: str, bars_per_window: int = 4,
                               min_window_seconds: float = 4.65,
                               adaptive_bars: bool = True):
    """Walk every item_X folder, extract notes, write per-item CSV + merged CSV."""
    item_dirs = sorted(
        glob.glob(os.path.join(root_dir, "item_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not item_dirs:
        raise FileNotFoundError(f"No item_* folders found under {root_dir}")

    all_notes: list[dict] = []
    all_measures: list[dict] = []
    all_windows: list[dict] = []
    all_meta: list[dict] = []

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
            notes, measures, meta = extract_notes_from_gp(gp_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        for m in measures:
            m["item"] = item_name

        windows = plan_bar_windows(
            measures, bars_per_window=bars_per_window,
            min_seconds=min_window_seconds, adaptive=adaptive_bars,
        )
        for w in windows:
            w["item"] = item_name

        stem = os.path.splitext(os.path.basename(gp_path))[0]

        _write_csv(os.path.join(item_dir, f"{stem}_notes.csv"), NOTE_FIELDNAMES, notes)
        _write_csv(os.path.join(item_dir, f"{stem}_measures.csv"), MEASURE_FIELDNAMES, measures)
        _write_csv(os.path.join(item_dir, f"{stem}_windows.csv"), WINDOW_FIELDNAMES, windows)
        with open(os.path.join(item_dir, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        short = sum(w["too_short"] for w in windows)
        varies = sum(w["bpm_varies"] for w in windows)
        used_bars = windows[0]["bars"] if windows else bars_per_window
        grew = " (grown from {})".format(bars_per_window) if used_bars != bars_per_window else ""
        print(f"  → {len(notes)} notes, {len(measures)} measures, "
              f"{len(windows)} windows of {used_bars} bars{grew} "
              f"(bpm {meta['bpm_values']}, {short} too short, {varies} with tempo changes)")

        all_notes.extend(notes)
        all_measures.extend(measures)
        all_windows.extend(windows)
        all_meta.append(meta)

    _write_csv(os.path.join(root_dir, "all_notes.csv"), NOTE_FIELDNAMES, all_notes)
    _write_csv(os.path.join(root_dir, "all_measures.csv"), MEASURE_FIELDNAMES, all_measures)
    _write_csv(os.path.join(root_dir, "all_windows.csv"), WINDOW_FIELDNAMES, all_windows)
    with open(os.path.join(root_dir, "all_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    # ── window-length summary: what the frame exporter has to work with ───────
    if all_windows:
        durs = sorted(w["duration_s"] for w in all_windows)
        n_short = sum(w["too_short"] for w in all_windows)
        bars_used = sorted({w["bars"] for w in all_windows})
        print(f"\n{len(all_windows)} windows, {bars_used} bars per window: "
              f"{durs[0]:.2f}-{durs[-1]:.2f} s (median {durs[len(durs)//2]:.2f} s)")
        if n_short:
            offenders = sorted({w["item"] for w in all_windows if w["too_short"]})
            print(f"  WARNING: {n_short} windows are <= {min_window_seconds} s and "
                  f"will break FretNet's slicing, in: {offenders[:10]}"
                  f"{' ...' if len(offenders) > 10 else ''}")
            print("  These items are too short to make a long-enough window even "
                  "at 16 bars. Either drop them, or lower FretNet's --num-frames "
                  "(150 frames needs only ~3.48 s).")
    print_technique_stats(all_notes)
    print(f"\nDone. {len(all_notes)} total notes → {os.path.join(root_dir, 'all_notes.csv')}")