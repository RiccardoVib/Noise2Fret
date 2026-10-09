import os
from pathlib import Path
from Noise2Fret_Aux_Losses.src.utils import find_folder_upward
from TimeTabExtraction import process_dataset_extraction
from Noise2Fret_Aux_Losses.data_preprocess.GOAT.AlignFrames import process_dataset_align

current_dir = Path(os.getcwd())
print(f"current_dir: {current_dir}")
files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)

FRAME_DURATION = 0.1

# Extract
ROOT_DIR_TRAIN = files_dir / "GOAT/train/"
process_dataset_extraction(ROOT_DIR_TRAIN)

ROOT_DIR_TEST = files_dir / "GOAT/test/"
process_dataset_extraction(ROOT_DIR_TEST)

# Align
process_dataset_align(ROOT_DIR_TRAIN, frame_duration=FRAME_DURATION)
process_dataset_align(ROOT_DIR_TEST, frame_duration=FRAME_DURATION, debug=False)

from MoveFiles import mirror_and_move

# ── Config ────────────────────────────────────────────────────────────────────
SPLITS = ["train", "test"]
EXTENSIONS = ["*.csv", "*.npy"]
# Set to True to copy instead of move (non-destructive)
COPY_ONLY = False

SRC_BASE = files_dir / "GOAT"
DST_BASE = files_dir / ("GOAT_processed_" + str(FRAME_DURATION))

for split in SPLITS:
    src = SRC_BASE / split
    dst = DST_BASE / split
    if not src.exists():
        print(f"[skip] {src} does not exist")
        continue
    print(f"\n[{split}]  {src}  →  {dst}")
    mirror_and_move(src, dst, copy_only=COPY_ONLY)
