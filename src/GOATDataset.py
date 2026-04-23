import os
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# Open-string MIDI pitches per string (same as DadaGPDataset)
# String index 0-based: s1=E2, s2=A2, s3=D3, s4=G3, s5=B3, s6=E4
OPEN_PITCHES = [40, 45, 50, 55, 59, 64]
PAD_FRET = -1
PAD_PC   = -1


# ── Constants ──────────────────────────────────
N_STRINGS   = 6

class GOATFrameDataset(Dataset):
    """
    Dataset returning TabCNN-style tablature targets.

    target / prev_target shape: (6, 21)
        - axis 0: string index (s1..s6)
        - axis 1: fret class   (0=muted, 1=open/fret0, 2=fret1, ..., 20=fret19)

    """
    PAD_NOTE_ID  = 0
    PAD_CHORD_ID = 0
    CHORD_LEN    = 7

    def __init__(
        self,
        root_dir: str,
        data_dir: str,
        max_events=None,
    ):
        self.root_dir         = root_dir
        self.data_dir         = data_dir
        self.fs               = 16000

        self.pad_note_id  = 0
        self.pad_chord_id = 0
        self.chord_len    = 7
        self.n_strings    = N_STRINGS

        self._npy_cache: dict[str, np.ndarray] = {}

        all_rows    = self._load_metadata(root_dir)
        self.meta   = pd.concat(all_rows, ignore_index=True)

        # ── max_events kept in case callers use it externally ──────────
        if max_events is None:
            self.max_events = int(
                self.meta["chords"]
                .dropna()
                .apply(lambda x: len(x.split("|")) if str(x).strip() and str(x).strip() != "nan" else 0)
                .max()
            )
        else:
            self.max_events = max_events

        # ── Compute fret_max ───────────────────────────
        fret_max = 0
        max_hand_span = 0
        for _, row in self.meta.iterrows():
            if pd.notna(row["chords"]):
                for group_str in str(row["chords"]).strip().split("|"):
                    tokens = self._extract_note(group_str)
                    frets  = self._encode_chord_frets(tokens)

                    active_frets = [f for f in frets if f != PAD_FRET and f > 0]  # exclude muted/open

                    if np.max(frets) > fret_max:
                        fret_max = np.max(frets)

                    if len(active_frets) >= 2:
                        span = max(active_frets) - min(active_frets)
                        if max_hand_span == 9:
                            print(group_str)
                        if span > max_hand_span:
                            max_hand_span = span

        print(f"max hand span: {max_hand_span}")
        print(f"fret max: {fret_max}")
        self.n_classes = fret_max + 2

        all_rows  = self._load_metadata(data_dir)
        self.meta = pd.concat(all_rows, ignore_index=True)

    # ── All existing helpers ────────────────────────────────────────
    def _load_metadata(self, root_dir):
        item_dirs = sorted(
            glob.glob(os.path.join(root_dir, "item_*")),
            key=lambda p: int(os.path.basename(p).split("_")[1]),
        )
        if not item_dirs:
            raise FileNotFoundError(f"No item_* folders under {root_dir}")

        all_rows = []
        global_offset = 0

        for item_dir in item_dirs:
            item = os.path.basename(item_dir)

            npy_files = glob.glob(os.path.join(item_dir, item + "_note_frames.npy"))
            meta_files = glob.glob(os.path.join(item_dir, item + "_note_frame_meta.csv"))

            if not npy_files or not meta_files:
                continue

            npy_path = npy_files[0]
            meta = pd.read_csv(meta_files[0])

            meta["_npy_path"] = npy_path
            meta["_global_idx"] = np.arange(global_offset, global_offset + len(meta))

            all_rows.append(meta)
            global_offset += len(meta)

        if not all_rows:
            raise RuntimeError("No valid items found.")
        return all_rows

    # ── token parsing helpers ────
    def _extract_note(self, group_str):
        tokens = []
        for token in group_str.split(" "):
            token = token.strip()
            if ":note:" in token:
                token = token.split(":note:", 1)[1]
            if token:
                tokens.append(token)
        return tokens

    def _encode_chord_frets(self, token_strings: list[str]) -> list[int]:
        frets = [PAD_FRET] * (self.CHORD_LEN)
        for t in token_strings:
            string_idx, fret, _ = self._parse_token(t)
            if string_idx is not None and fret != PAD_FRET:
                frets[string_idx] = fret
        return frets[:-1]  # drop bend slot for frets

    def _encode_chord_pcs(self, token_strings: list[str]) -> list[int]:
        """Token strings → PCs by string order (1→0, 2→1, ...) → PAD_PC-padded."""
        pcs = [PAD_PC] * (self.CHORD_LEN)  # init all strings empty
        for t in token_strings:
            string_idx, _, pc = self._parse_token(t)
            if string_idx is not None and pc != PAD_PC:
                pcs[string_idx] = pc
        return pcs[:-1]

    @staticmethod
    def _parse_token(token_str: str):
        """
        Parse 'sX:fY' → (string_idx 0-based, fret, pitch_class).
        Returns (None, PAD_FRET, PAD_PC) for non-note tokens.
        Format: <instrument>:note:s<string>:f<fret>
        e.g.   s1:f5  → string=0 (1-based→0-based), fret=5
        """
        parts = token_str.split(":")
        if len(parts) == 2 and parts[0].startswith("s") and parts[1].startswith("f"):
            try:
                string_idx = int(parts[0][1:]) - 1   # 1-based → 0-based
                fret       = int(parts[1][1:])
                if 0 <= string_idx < len(OPEN_PITCHES):
                    pc = (OPEN_PITCHES[string_idx] + fret) % 12
                    return string_idx, fret, pc
            except ValueError:
                pass
        return None, PAD_FRET, PAD_PC

    def _parse_chords(self, row) -> list[list[str]]:
        """Helper: split chords column into groups of tokens."""
        chords_raw = str(row["chords"]).strip()
        if not chords_raw or chords_raw == "nan":
            return []
        return [group.split() for group in chords_raw.split("|")]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]

        # ── Audio ─────────────────────────────────────────
        npy_path = row["_npy_path"]
        if npy_path not in self._npy_cache:
            self._npy_cache[npy_path] = np.load(npy_path, mmap_mode="r")

        audio = torch.from_numpy(
            self._npy_cache[npy_path][int(row["frame_idx"])].copy()
        )
        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]
        if audio.shape[1] > 1:
            audio = torch.mean(audio, dim=-1, keepdim=True)
        audio = torchaudio.functional.resample(audio.T, 44100, self.fs).T

        # ── Target: (6, F) one-hot matrix ────────────────────────────
        target = self._build_tab_target(row)          # (6, F) float32

        # ── Previous target ───────────────────────────────────────────
        is_first_frame = (
            idx == 0
            or self.meta.iloc[idx - 1]["_npy_path"] != npy_path
        )
        if is_first_frame:
            prev_target = []
            while len(prev_target) < self.max_events:
                prev_target.append(self._empty_tab_target())   # all strings muted
            prev_target =  torch.stack(prev_target, dim=0)
        else:
            prev_target = self._build_tab_target(self.meta.iloc[idx - 1])

        return audio, target, prev_target

    # ── target helpers ────────────────────────────────────────────────────

    def _fret_to_class(self, fret: int) -> int:
        """
        Map raw fret number to class index.
            fret  0  (open)  → class 1
            fret  1          → class 2
            ...
            fret 19          → class 20
        Muted strings are handled separately (class 0).
        """
        return fret + 1   # shift by 1; class 0 is reserved for muted

    def _empty_tab_target(self) -> torch.Tensor:
        """Returns a (6, 21) tensor with class-0 (muted) set for all strings."""
        tab = torch.zeros(N_STRINGS, self.n_classes, dtype=torch.float32)
        tab[:, 0] = 1.0   # all strings muted
        return tab

    def _build_tab_target(self, row) -> torch.Tensor:
        """
        Parse the 'chords' column and build a (6, 21) one-hot matrix.

        Multiple chord groups in one frame are merged by last-write-wins
        (very rare in guitar; a single frame almost always has one group).

        Layout:
            tab[string_idx, 0]       = 1  → string muted
            tab[string_idx, fret+1]  = 1  → string plays <fret>
        """
        frames, frets, pcs = [], [], []
        chord_groups = self._parse_chords(row)
        for chord_group in chord_groups:
            frame = self._empty_tab_target()  # start with all muted
            tokens = self._extract_note(" ".join(chord_group))

            for token in tokens:
                string_idx, fret, _ = self._parse_token(token)
                if string_idx is None:
                    continue                     # bend or unrecognised token
                cls = self._fret_to_class(fret)

                frame[string_idx, :] = 0.0        # clear previous one-hot
                frame[string_idx, cls] = 1.0      # set active class
            frames.append(frame)

        # Pad with muted frames if not enough
        while len(frames) < self.max_events:
            frames.append(self._empty_tab_target())  # (6, F) all-muted

        return torch.stack(frames, dim=0)            # (T, 6, 24)

    # ── Decode helper for the new target format ───────────────────────────────

    def decode_tab_target(self, tab: torch.Tensor) -> list[str]:
        """
        (6, 21) one-hot → human-readable list of 'sX:fY' / 'sX:muted' strings.

        Args:
            tab: (6, 21) float or long tensor

        Returns:
            ['s1:f5', 's2:muted', 's3:f0', ...]
        """
        result = []
        for s_idx in range(N_STRINGS):
            cls = int(tab[s_idx].argmax())
            if cls == 0:
                result.append(f"s{s_idx+1}:muted")
            else:
                result.append(f"s{s_idx+1}:f{cls-1}")

        active = [t for t in result if "muted" not in t]

        return active