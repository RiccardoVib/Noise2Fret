"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import os
import glob
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from OnsetsHead import AUDIO_HOP, build_onset_frame_target, frame_validity_mask


OPEN_PITCHES = [64, 59, 55, 50, 45, 40]
PAD_FRET = -1
PAD_PC = -1


class GOATFrameDataset(Dataset):

    PAD_TOKEN = "PAD"
    MUTE_FMT = "s{}MUTE"
    FULL_FRET_FMT = "s{}f{}"

    def __init__(
        self,
        root_dir: str,
        data_dir: str,
        max_events=None,
        random_crop_lengths: bool = True,
        onset_sigma_frames: float = 1.0,
    ):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.fs = 16000
        self.audio_length = 16000
        self.min_audio_length = 1600 * 2
        self.random_crop_lengths = bool(random_crop_lengths)
        self.n_strings = 6
        self.chord_len = 7

        self.audio_hop = AUDIO_HOP
        self.n_onset_frames = self.audio_length // self.audio_hop
        self.window_sec = self.audio_length / self.fs
        self.onset_sigma_frames = float(onset_sigma_frames)

        self._npy_cache: dict[str, np.ndarray] = {}

        metadata_root = self._load_metadata(root_dir)
        self.meta = pd.concat(metadata_root, ignore_index=True)

        if max_events is None:
            self.max_events = self._compute_max_number_of_events()
        else:
            self.max_events = int(max_events)

        max_fret, max_hand_span = self._compute_max_fret()
        self.max_fret = max_fret
        print(f"max hand span: {max_hand_span}")
        print(f"fret max: {max_fret}")

        self.vocab = self._build_vocab_full(max_fret)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab[self.PAD_TOKEN]
        self.seq_len = self.max_events * self.n_strings

        data_metadata = self._load_metadata(data_dir)
        self.meta = pd.concat(data_metadata, ignore_index=True)

    # ------------------------------------------------------------------
    # Metadata and vocabulary
    # ------------------------------------------------------------------

    def _load_metadata(self, root_dir: str) -> List[pd.DataFrame]:
        item_dirs = sorted(
            [
                p
                for p in glob.glob(
                    os.path.join(root_dir, "**", "item_*"), recursive=True
                )
                if os.path.isdir(p)
            ],
            key=lambda p: int(os.path.basename(p).split("_")[1]),
        )

        if not item_dirs:
            raise FileNotFoundError(f"No item_* folders under {root_dir}")

        all_rows = []
        global_offset = 0

        for item_dir in item_dirs:
            item = os.path.basename(item_dir)
            npy_files = glob.glob(
                os.path.join(item_dir, item + "_note_frames.npy")
            )
            meta_files = glob.glob(
                os.path.join(item_dir, item + "_note_frame_meta.csv")
            )

            if not npy_files or not meta_files:
                continue

            npy_path = npy_files[0]
            meta = pd.read_csv(meta_files[0])
            meta["_npy_path"] = npy_path
            meta["_global_idx"] = np.arange(
                global_offset, global_offset + len(meta)
            )
            meta = meta.dropna(subset=["chords"]).reset_index(drop=True)

            all_rows.append(meta)
            global_offset += len(meta)

        if not all_rows:
            raise RuntimeError("No valid items found.")

        return all_rows

    def _build_vocab_full(self, max_fret: int) -> dict[str, int]:
        vocab = {self.PAD_TOKEN: 0}
        next_id = 1

        for string_idx in range(1, self.n_strings + 1):
            vocab[self.MUTE_FMT.format(string_idx)] = next_id
            next_id += 1

        for string_idx in range(1, self.n_strings + 1):
            for fret in range(max_fret + 1):
                vocab[self.FULL_FRET_FMT.format(string_idx, fret)] = next_id
                next_id += 1

        return vocab

    # ------------------------------------------------------------------
    # Chord parsing and metadata statistics
    # ------------------------------------------------------------------

    def _compute_max_number_of_events(self) -> int:
        return int(
            self.meta["chords"]
            .dropna()
            .apply(
                lambda x: len(x.split("|"))
                if str(x).strip() and str(x).strip() != "nan"
                else 0
            )
            .max()
        )

    def _compute_max_fret(self) -> Tuple[int, int]:
        max_fret = 0
        max_hand_span = 0

        for _, row in self.meta.iterrows():
            for group_str in str(row["chords"]).strip().split("|"):
                tokens = self._extract_note(group_str)
                frets = self._encode_chord_frets(tokens)

                valid_frets = [f for f in frets if f != PAD_FRET and f >= 0]
                if valid_frets:
                    max_fret = max(max_fret, max(valid_frets))

                active_frets = [f for f in frets if f != PAD_FRET and f > 0]
                if len(active_frets) >= 2:
                    max_hand_span = max(
                        max_hand_span,
                        max(active_frets) - min(active_frets),
                    )

        return max_fret, max_hand_span

    def _extract_note(self, group_str: str) -> List[str]:
        tokens = []
        for token in group_str.split():
            token = token.strip()
            if ":note:" in token:
                token = token.split(":note:", 1)[1]
            if token:
                tokens.append(token)
        return tokens

    @staticmethod
    def _parse_token(token_str: str) -> Tuple[int | None, int, int]:
        parts = token_str.split(":")
        if len(parts) != 2 or not parts[0].startswith("s"):
            return None, PAD_FRET, PAD_PC

        try:
            string_idx = int(parts[0][1:]) - 1
            fret_part = parts[1]
            if not fret_part.startswith("f"):
                return None, PAD_FRET, PAD_PC
            fret = int(fret_part[1:])
        except ValueError:
            return None, PAD_FRET, PAD_PC

        if not 0 <= string_idx < len(OPEN_PITCHES):
            return None, PAD_FRET, PAD_PC

        pc = (OPEN_PITCHES[string_idx] + fret) % 12
        return string_idx, fret, pc

    def _encode_chord_frets(self, token_strings: List[str]) -> List[int]:
        frets = [PAD_FRET] * self.chord_len

        for token in token_strings:
            string_idx, fret, _ = self._parse_token(token)
            if string_idx is not None and fret != PAD_FRET:
                frets[string_idx] = fret

        return frets[:-1]

    def _encode_chord_pcs(self, token_strings: List[str]) -> List[int]:
        pcs = [PAD_PC] * self.chord_len

        for token in token_strings:
            string_idx, _, pc = self._parse_token(token)
            if string_idx is not None and pc != PAD_PC:
                pcs[string_idx] = pc

        return pcs[:-1]

    def _parse_chords(self, row) -> List[List[str]]:
        chords_raw = str(row["chords"]).strip()
        if not chords_raw or chords_raw == "nan":
            return []
        return [group.split() for group in chords_raw.split("|")]

    def _parse_onsets(self, row) -> List[float]:
        raw = row.get("chord_onsets", None)
        text = str(raw).strip()
        if not text or text == "nan":
            return []
        return [float(x.strip()) for x in text.split(";") if x.strip()]

    def _pair_events_with_onsets(
        self, row
    ) -> List[Tuple[float, List[str]]]:
        chord_groups = self._parse_chords(row)
        chord_onsets = self._parse_onsets(row)
        return list(zip(chord_onsets, chord_groups))

    # ------------------------------------------------------------------
    # Vocabulary token encoding
    # ------------------------------------------------------------------

    def _event_group_to_tokens(self, group: List[str]) -> List[int]:
        event_tokens = [
            self.vocab[self.MUTE_FMT.format(string_idx + 1)]
            for string_idx in range(self.n_strings)
        ]

        for token in self._extract_note(" ".join(group)):
            string_idx, fret, _ = self._parse_token(token)
            if string_idx is None or fret == PAD_FRET:
                continue

            token_name = self.FULL_FRET_FMT.format(string_idx + 1, fret)
            if token_name not in self.vocab:
                raise ValueError(
                    f"Token {token_name!r} is absent from the vocabulary. "
                    f"Maximum fret is {self.max_fret}."
                )

            event_tokens[string_idx] = self.vocab[token_name]

        return event_tokens

    def _events_to_token_ids(
        self, event_groups: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = []

        for group in event_groups[: self.max_events]:
            flat.extend(self._event_group_to_tokens(group))

        n_real = len(flat)
        flat.extend([self.pad_id] * (self.seq_len - n_real))
        flat = flat[: self.seq_len]

        token_ids = torch.tensor(flat, dtype=torch.long)
        pad_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        pad_mask[n_real:] = True

        return token_ids, pad_mask

    def _row_to_token_ids(
        self, row
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._events_to_token_ids(self._parse_chords(row))

    def decode_token_ids(self, token_ids: torch.Tensor) -> List[List[str]]:
        token_ids = token_ids.detach().cpu().tolist()
        tokens = [self.id_to_token[int(token_id)] for token_id in token_ids]

        return [
            tokens[i : i + self.n_strings]
            for i in range(0, len(tokens), self.n_strings)
        ]

    # ------------------------------------------------------------------
    # Audio and window construction
    # ------------------------------------------------------------------

    def _load_audio(self, row) -> torch.Tensor:
        npy_path = row["_npy_path"]

        if npy_path not in self._npy_cache:
            self._npy_cache[npy_path] = np.load(npy_path, mmap_mode="r")

        audio = torch.from_numpy(
            self._npy_cache[npy_path][int(row["frame_idx"])].copy()
        ).float()

        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] > 1:
            audio = torch.mean(audio, dim=-1, keepdim=True)

        audio = torchaudio.functional.resample(
            audio.T, 44100, self.fs
        ).T
        audio = audio / audio.abs().max().clamp(min=1e-8)
        return audio.contiguous()

    def _build_streaming_window(self, audio: torch.Tensor):
        n_channels = audio.shape[1]

        if self.random_crop_lengths:
            effective_len = random.randint(
                self.min_audio_length, self.audio_length
            )
        else:
            effective_len = min(audio.shape[0], self.audio_length)

        crop = audio[:effective_len]
        output = torch.zeros(
            self.audio_length, n_channels, dtype=audio.dtype
        )
        valid_context_mask = torch.zeros(
            self.audio_length, 1, dtype=audio.dtype
        )

        output[:effective_len] = crop
        valid_context_mask[:effective_len] = 1.0

        crop_end_sec = effective_len / self.fs
        return output, valid_context_mask, effective_len, crop_end_sec

    # ------------------------------------------------------------------
    # Dataset targets
    # ------------------------------------------------------------------

    def _events_in_window(self, row, crop_end_sec: float):
        paired = self._pair_events_with_onsets(row)
        paired = sorted(
            [pair for pair in paired if pair[0] < crop_end_sec],
            key=lambda pair: pair[0],
        )
        return (
            [group for _, group in paired],
            [onset for onset, _ in paired],
        )

    def pad_tab_tokens(self):
        token_ids = torch.full(
            (self.seq_len,), self.pad_id, dtype=torch.long
        )
        pad_mask = torch.ones(self.seq_len, dtype=torch.bool)
        return token_ids, pad_mask

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]

        audio = self._load_audio(row)
        (
            audio,
            valid_context_mask,
            effective_audio_length,
            crop_end_sec,
        ) = self._build_streaming_window(audio)

        event_groups, event_onsets_rel = self._events_in_window(
            row, crop_end_sec
        )
        tab_tokens, tab_mask = self._events_to_token_ids(event_groups)

        onsets = event_onsets_rel[: self.max_events]
        onset_frames = build_onset_frame_target(
            onsets,
            self.n_onset_frames,
            self.window_sec,
            self.onset_sigma_frames,
        )
        onset_frame_mask = frame_validity_mask(
            effective_audio_length,
            self.n_onset_frames,
            self.audio_hop,
        )

        onset_times = torch.zeros(
            self.max_events, dtype=torch.float32
        )
        onset_times_mask = torch.zeros(
            self.max_events, dtype=torch.bool
        )

        for event_idx, onset in enumerate(onsets):
            onset_times[event_idx] = min(
                max(float(onset) / self.window_sec, 0.0), 1.0
            )
            onset_times_mask[event_idx] = True

        npy_path = row["_npy_path"]
        is_first_frame = (
            idx == 0
            or self.meta.iloc[idx - 1]["_npy_path"] != npy_path
        )

        if is_first_frame:
            prev_tab_tokens, prev_tab_mask = self.pad_tab_tokens()
        else:
            prev_tab_tokens, prev_tab_mask = self._row_to_token_ids(
                self.meta.iloc[idx - 1]
            )

        return {
            "audio": audio,
            "tab_tokens": tab_tokens,
            "tab_mask": tab_mask,
            "prev_tab_tokens": prev_tab_tokens,
            "prev_tab_mask": prev_tab_mask,
            "idx": int(idx),
            "valid_context_mask": valid_context_mask,
            "onset_frames": onset_frames,
            "onset_frame_mask": onset_frame_mask,
            "onset_times": onset_times,
            "onset_times_mask": onset_times_mask,
        }
