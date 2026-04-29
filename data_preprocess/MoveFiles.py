import os
import shutil
from pathlib import Path
from Code.Utils.utils import find_folder_upward

# ── Config ────────────────────────────────────────────────────────────────────
SPLITS = ["train", "test"]
EXTENSIONS = ["*.csv", "*.npy"]
# Set to True to copy instead of move (non-destructive)
COPY_ONLY = False
# ─────────────────────────────────────────────────────────────────────────────

def mirror_and_move(src_root: str, dst_root: str, copy_only: bool = False):
    """
    For every item_XX folder under src_root:
      - Create a matching empty folder under dst_root
      - Move (or copy) all .csv and .npy files into it
    Also moves root-level merged files (all_notes.csv, all_note_frame_meta.csv).
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    action = shutil.copy2 if copy_only else shutil.move
    action_name = "Copied" if copy_only else "Moved"

    # 1. Per-item files
    item_dirs = sorted(
        src_root.glob("item*"),
        key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else 0
    )

    if not item_dirs:
        print(f"  [!] No item_XX folders found under {src_root}")
        return

    for item_dir in item_dirs:
        if not item_dir.is_dir():
            continue

        out_dir = dst_root / item_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for pattern in EXTENSIONS:
            for src_file in item_dir.glob(pattern):
                dst_file = out_dir / src_file.name
                action(str(src_file), str(dst_file))
                print(f"  {action_name}: {src_file.relative_to(src_root.parent)} "
                      f"→ {dst_file.relative_to(dst_root.parent)}")
                moved += 1

        if moved == 0:
            print(f"  [~] No .csv/.npy found in {item_dir.name}")

    # 2. Root-level merged files (all_notes.csv, all_note_frame_meta.csv)
    for pattern in EXTENSIONS:
        for src_file in src_root.glob(pattern):
            dst_file = dst_root / src_file.name
            action(str(src_file), str(dst_file))
            print(f"  {action_name} (root): {src_file.name} → {dst_root.name}/{src_file.name}")

    print(f"\n  Done: {src_root.name} → {dst_root.name}")
