import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


STRING_NAMES = ["E", "A", "D", "G", "B", "e"]


# --------------------------------------------------------------------------- #
# split helper
# --------------------------------------------------------------------------- #
def split_files(files, test_player="00", val_ratio=0.1, seed=0):
    """
    GuitarSet chunk names look like  '00_BN1-129-Eb_comp_03.npz'
                                      ^^ player        ^^ 4-bar chunk index
    Chunks of one recording must stay on the same side of the split, otherwise
    neighbouring (and, with 100 ms windows, near-identical) audio leaks from
    train into validation.  Returns (train, val, test) lists of paths.
    """
    files = sorted(files)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    test = [f for f in files if stem(f).startswith(f"{test_player}_")]
    dev = [f for f in files if not stem(f).startswith(f"{test_player}_")]

    recordings = sorted({stem(f).rsplit("_", 1)[0] for f in dev})
    rng = random.Random(seed)
    rng.shuffle(recordings)
    n_val = max(1, int(round(len(recordings) * val_ratio)))
    val_rec = set(recordings[:n_val])

    train = [f for f in dev if stem(f).rsplit("_", 1)[0] not in val_rec]
    val = [f for f in dev if stem(f).rsplit("_", 1)[0] in val_rec]
    return train, val, test


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class WindowedGuitarSet(Dataset):
    """
    files          : list of split .npz paths (output of midi_to_numpy.split_save)
    sr, hop        : down_sampling_rate and hop_length from config.yaml
                     (frame i is centred at time i*hop/sr, as in midi_to_numpy.py)
    win_s          : window length in seconds; hop between windows == window length
                     (non-overlapping, grid anchored at the start of each chunk).
                     A trailing remainder < one window is dropped.
    snap_window    : True (default) -> the window is rounded to a whole number of
                     feature frames, so EVERY window has exactly the same number of
                     frames and samples.  0.1 s at 22050/512 is 4.31 frames, which
                     would give 4 or 5 frames depending on the window; snapping gives
                     4 frames = 2048 samples = 92.9 ms.  Real length: `win_samples`,
                     `frames_per_window`, `win_s_actual`.  False -> exact win_s,
                     frame count varies (4 or 5).
    mode           : "tab" -> labels from frame_tab ; "F0" -> frame_F0 (sounding only)
    target         : "events" (clustered chord events) or "sounding"
    cluster_frames : onsets within this many frames of a cluster's FIRST onset join
                     it.  2 frames ~ 46 ms at 22050/512.  0 = only onsets on the
                     very same frame are merged.
    merge          : how the clustered events of ONE window are simplified.
                     "notes"  (default) merge NOTES: a note (string, fret) that already
                              appeared in an earlier event of the window is removed from
                              the later ones, and events left empty vanish.  A chord that
                              repeats with only one string changed therefore becomes
                              {E:3, A:5}, {E:3, A:7}  ->  {E:3, A:5}, {A:7}: only the
                              string that changed is separated.
                     "chords" merge CHORDS: events with exactly the same note set are
                              merged (A,B,A,B -> A,B); {E:3,A:5} != {E:3,A:7}.
                     "none"   keep every clustered event.
                     Event time = first onset of the cluster the event came from.
    max_events     : number of event slots per window.  None (default) -> the largest
                     count found in `files` after merging.  An int is used as is
                     (e.g. pass the train value to val/test).
    mute_index     : index of the 'not played' class (-1 = last, midi_to_numpy
                     layout; 0 if class 0 = muted).
    drop_empty     : drop windows without any event (target='events') / without
                     any sounding note (target='sounding').
    preload        : keep the needed arrays in RAM (float32).  Recommended: otherwise
                     every 100 ms window decompresses a whole 4-bar .npz.
    return_event_times : append event onset times (s from window start) to the tuple.

    Attributes after construction: n_classes, mute_index, max_events, n_over_cap,
    over_cap_examples, event_hist (events per window as clustered), distinct_hist
    (after merging).
    """

    def __init__(self, files, sr, hop, win_s=0.1, mode="tab", target="events",
                 cluster_frames=2, max_events=None, input_feature_type="cqt",
                 mute_index=-1, drop_empty=False, preload=True,
                 return_event_times=False, snap_window=True, merge="notes"):
        assert target in ("events", "sounding")
        assert merge in ("notes", "chords", "none")
        self.files = list(files)
        self.sr, self.hop, self.win_s = int(sr), int(hop), float(win_s)
        self.mode = mode
        self.target = "sounding" if mode == "F0" else target
        self.cluster_frames = int(cluster_frames)
        self.merge = merge
        self._cap = None if max_events is None else int(max_events)
        self.input_feature_type = input_feature_type
        self.drop_empty = drop_empty
        self.preload = preload
        self.return_event_times = return_event_times
        self.snap_window = snap_window
        if snap_window:
            self.frames_per_window = max(1, int(round(self.win_s * self.sr / self.hop)))
            self.win_samples = self.frames_per_window * self.hop
        else:
            self.frames_per_window = None                      # 4 or 5, varies
            self.win_samples = int(round(self.win_s * self.sr))
        self.win_s_actual = self.win_samples / self.sr
        self.n_strings = 6

        self.n_classes = None
        self.mute_index = mute_index
        self._cache = {}
        self.index = []                  # (file idx, window k)
        self.event_hist = Counter()      # events per window (all windows, before drop_empty)

        self.distinct_hist = Counter()   # events per window AFTER merging identical ones
        self.n_over_cap = 0              # windows still above max_events after that
        self.over_cap_examples = []      # one text line per such window
        for fi in range(len(self.files)):
            d = self._load(fi)
            for k in range(len(d["audio"]) // self.win_samples):
                n_raw = d["n_raw"].get(k, 0)
                n_dist = d["n_distinct"].get(k, 0)
                self.event_hist[n_raw] += 1
                self.distinct_hist[n_dist] += 1
                if k in d["over"]:
                    self.n_over_cap += 1
                    self.over_cap_examples.append(
                        self._describe_window(self.files[fi], k, d["over"][k]))
                    continue                         
                if self.drop_empty:
                    keep = n_dist > 0 if self.target == "events" else self._window_has_notes(d, k)
                    if not keep:
                        continue
                self.index.append((fi, k))
            if not preload:
                self._cache.pop(fi, None)

        found_max = max(self.distinct_hist) if self.distinct_hist else 1
        if self.target == "sounding":
            self.max_events = 1
        else:
            self.max_events = self._cap if self._cap is not None else max(found_max, 1)

        if self.target == "events" and self.mode == "tab":
            self._print_summary()

    def _print_summary(self):
        frames = self.frames_per_window if self.snap_window else "4-5 (varies)"
        print(f"[WindowedGuitarSet] window {self.win_s_actual * 1000:.1f} ms = {self.win_samples} "
              f"samples, frames/window {frames} | {len(self.index)} windows | "
              f"max_events={self.max_events}")
        print(f"   events/window as clustered         : {sorted(self.event_hist.items())}")
        print(f"   {'after merge=' + repr(self.merge):<35}: {sorted(self.distinct_hist.items())}")
        if self.n_over_cap:
            total = sum(self.distinct_hist.values())
            action = "REMOVED from the dataset"
            print(f"   WARNING: {self.n_over_cap} windows ({self.n_over_cap / total:.2%}) have more "
                  f"than {self.max_events} {self._what()} -> {action}. First examples:")
            for line in self.over_cap_examples[:5]:
                print("     " + line)
            print("   all of them: dataset.write_over_cap_report(path)")

    def write_over_cap_report(self, path):
        """Write one line per window that has more than max_events different note sets."""
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{self.n_over_cap} windows with more than {self.max_events} "
                     f"{self._what()}\n")
            fh.write("\n".join(self.over_cap_examples) + "\n")

    def _what(self, plural_phrase=True):
        if self.merge != "none":
            return "DIFFERENT events" if plural_phrase else "different"
        return "events"

    def _fret(self, c):
        return int(c) if self.mute_index == self.n_classes - 1 else int(c) - 1

    def _describe_window(self, path, k, evs):
        parts = []
        for f, cls in evs:
            notes = " ".join(f"{STRING_NAMES[s]}:{self._fret(c)}"
                             for s, c in enumerate(cls) if c != self.mute_index)
            parts.append(f"[frame {f}: {notes}]")
        return f"{os.path.basename(path)} window {k} ({len(evs)} {self._what(False)}): " + " ".join(parts)

    # ---- file access ------------------------------------------------------ #
    def _load(self, fi):
        if fi in self._cache:
            return self._cache[fi]
        z = np.load(self.files[fi])
        feat_key = "cqt" if self.input_feature_type == "cqt" else "mel_spec"
        gt_key = "frame_tab" if self.mode == "tab" else "frame_F0"
        d = {
            "feat": z[feat_key].astype(np.float32),
            "stft": z["stft"].astype(np.float32),
            "sf": z["sf"].astype(np.float32),
            "b": z["b"].astype(np.float32),
            "audio": z["audio"].astype(np.float32),
            "frame_gt": z[gt_key].astype(np.float32),
            "bpm": np.float32(z["tempo"]),
        }
        n = d["frame_gt"].shape[0]
        assert abs(len(d["audio"]) - n * self.hop) <= self.hop, (
            f"{self.files[fi]}: {len(d['audio'])} samples vs {n} frames * hop {self.hop}; "
            "check sr / hop against config.yaml")

        if self.n_classes is None:
            self.n_classes = d["frame_gt"].shape[-1]
            if self.mode == "tab" and self.mute_index < 0:
                self.mute_index = self.n_classes + self.mute_index

        d["events"] = {}
        if self.mode == "tab":
            onset_cls = z["frame_tab_onset"].argmax(-1)                     # (T, 6)
            n_win = len(d["audio"]) // self.win_samples
            d["events"] = self._cluster_events(onset_cls, n_win)
        d["n_raw"] = {k: len(v) for k, v in d["events"].items()}
        d["n_distinct"], d["over"] = dict(d["n_raw"]), {}
        if self.mode == "tab" and self.target == "events":
            for k, evs in list(d["events"].items()):
                evs = self._merge_events(evs)
                d["n_distinct"][k] = len(evs)
                if self._cap is not None and len(evs) > self._cap:
                    d["over"][k] = evs                      
                d["events"][k] = evs

        if self.preload:
            self._cache[fi] = d
        return d

    # ---- chord clustering ------------------------------------------------- #
    def _cluster_events(self, onset_cls, n_win):
        """
        onset_cls : (T, 6) class id at each frame's onset label (mute_index = no onset)
        returns   : {window k: [(start_frame, class_ids(6,)), ...]} time ordered
        """
        mute, tol = self.mute_index, self.cluster_frames
        is_on = onset_cls != mute
        clusters, cur = [], None
        for f in np.flatnonzero(is_on.any(axis=1)):
            strings = np.flatnonzero(is_on[f])
            fits = (cur is not None and f - cur[0] <= tol
                    and not any(cur[1][s] != mute for s in strings))
            if not fits:                                   # start a new cluster
                cur = (int(f), np.full(self.n_strings, mute, dtype=np.int64))
                clusters.append(cur)
            for s in strings:
                cur[1][s] = onset_cls[f, s]

        events = defaultdict(list)
        for f, cls in clusters:
            k = (f * self.hop) // self.win_samples         # window containing frame centre
            if k < n_win:
                events[k].append((f, cls))
        return dict(events)

    # ---- merge events inside one window ----------------------------------- #
    def _merge_events(self, evs):
        if self.merge == "notes":
            return self._merge_notes(evs)
        if self.merge == "chords":
            return self._dedupe_events(evs)
        return evs

    def _merge_notes(self, evs):
        """
        Merge NOTES, not chords.  Walk the window's events in time order; a note
        (string, fret) already placed in an earlier event is removed from the later
        ones; an event left with no note disappears.

            {E:3, A:5}  {E:3, A:7}   ->   {E:3, A:5}  {A:7}
            {E:3} {E:4} {E:3} {E:4}  ->   {E:3} {E:4}
        """
        mute = self.mute_index
        seen, out = set(), []
        for f, cls in evs:
            ids = np.full(self.n_strings, mute, dtype=np.int64)
            for s in range(self.n_strings):
                c = int(cls[s])
                if c != mute and (s, c) not in seen:
                    seen.add((s, c))
                    ids[s] = c
            if (ids != mute).any():
                out.append((f, ids))
        return out

    @staticmethod
    def _dedupe_events(evs):
        """
        Events with the same note set (class id on every string) become one event at
        the first onset.  Order of first occurrence is kept: A,A,B,A -> A,B.
        """
        seen, out = set(), []
        for f, cls in evs:
            key = tuple(int(c) for c in cls)
            if key not in seen:
                seen.add(key)
                out.append((f, cls))
        return out

    # ---- window geometry -------------------------------------------------- #
    def _frame_range(self, k, n_frames):
        """Frames whose centre i*hop falls in [k*win, (k+1)*win)  (ceil division)."""
        f0 = -(-(k * self.win_samples) // self.hop)
        f1 = -(-((k + 1) * self.win_samples) // self.hop)
        return f0, min(f1, n_frames)

    def _window_has_notes(self, d, k):
        f0, f1 = self._frame_range(k, d["frame_gt"].shape[0])
        sl = d["frame_gt"][f0:f1]
        if self.mode == "tab":
            return bool(np.delete(sl, self.mute_index, axis=-1).any())
        return bool(sl.any())

    # ---- targets ---------------------------------------------------------- #
    def _event_targets(self, d, k):
        """-> note_gt (E, 6, C) one-hot, n_real, times (E,) seconds from window start."""
        E, C, mute = self.max_events, self.n_classes, self.mute_index
        evs = d["events"].get(k, [])[:E]
        ids = np.full((E, self.n_strings), mute, dtype=np.int64)     # padding = all mute
        times = np.full(E, -1.0, dtype=np.float32)
        for j, (f, cls) in enumerate(evs):
            ids[j] = cls
            times[j] = (f * self.hop - k * self.win_samples) / self.sr
        note_gt = np.eye(C, dtype=np.float32)[ids]                    # (E, 6, C)
        return note_gt, len(evs), times

    def _sounding_summary(self, frame_slice):
        """
        tab: string sounds in ANY frame -> the fret sounding in most frames
             (ties -> lowest fret); else 'not played'.   F0: max over frames.
        """
        if self.mode == "F0":
            return frame_slice.max(axis=0, keepdims=True)

        C = frame_slice.shape[-1]
        counts = frame_slice.sum(axis=0)
        sounding = np.delete(counts, self.mute_index, axis=-1)
        fret_ids = np.delete(np.arange(C), self.mute_index)
        out = np.zeros((1, self.n_strings, C), dtype=np.float32)
        for s in range(self.n_strings):
            if sounding[s].sum() > 0:
                out[0, s, fret_ids[sounding[s].argmax()]] = 1.0
            else:
                out[0, s, self.mute_index] = 1.0
        return out

    # ---- torch API -------------------------------------------------------- #
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, k = self.index[i]
        d = self._load(fi)
        f0, f1 = self._frame_range(k, d["frame_gt"].shape[0])
        s0 = k * self.win_samples
        frame_gt = d["frame_gt"][f0:f1]

        if self.target == "events":
            note_gt, n_real, times = self._event_targets(d, k)
        else:
            note_gt = self._sounding_summary(frame_gt)
            n_real = note_gt.shape[0]
            times = np.full(1, -1.0, dtype=np.float32)

        item = (d["feat"][f0:f1], frame_gt, note_gt, f1 - f0, n_real, d["bpm"],
                d["stft"][f0:f1], d["sf"][f0:f1], d["b"][f0:f1],
                d["audio"][s0:s0 + self.win_samples], self.win_samples)
        return item + (times,) if self.return_event_times else item


# --------------------------------------------------------------------------- #
# collate
# --------------------------------------------------------------------------- #
def window_collate(batch):
   
    cols = list(zip(*batch))
    frame_len = np.asarray(cols[3])
    order = np.argsort(-frame_len, kind="stable")

    def take(c):
        return [cols[c][j] for j in order]

    def pad_stack(arrs):
        m = max(a.shape[0] for a in arrs)
        return np.stack([np.pad(a, [(0, m - a.shape[0])] + [(0, 0)] * (a.ndim - 1))
                         for a in arrs])

    out = (
        torch.from_numpy(pad_stack(take(0))).float(),            # input features
        torch.from_numpy(pad_stack(take(1))).float(),            # frame_gt
        torch.from_numpy(np.stack(take(2))).float(),             # note_gt (B, E, 6, C)
        torch.from_numpy(frame_len[order]),
        torch.from_numpy(np.asarray(take(4))),                   # note_len (real events)
        torch.from_numpy(np.asarray(take(5))),                   # bpm
        torch.from_numpy(pad_stack(take(6))).float(),            # stft
        torch.from_numpy(pad_stack(take(7))).float(),            # sf
        torch.from_numpy(pad_stack(take(8))).float(),            # b
        torch.from_numpy(pad_stack(take(9))).float(),            # audio
        torch.from_numpy(np.asarray(take(10))),                  # audio_len
    )
    if len(cols) > 11:
        out = out + (torch.from_numpy(np.stack(take(11))),)      # event_times
    return out
