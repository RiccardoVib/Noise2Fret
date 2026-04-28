from pathlib import Path
import numpy as np
import torch

from tab_metrics import tab_metrics, print_tab_metrics
from diffusion_training import vectors_to_text_token, print_results

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG ← edit this block only
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent

_MODELS_ROOT = _SCRIPT_DIR.parent.parent.parent / "TrainedModels" / ""
MODEL_NAMEs = [
    ""
]

SPLITS = ["test"]


# ══════════════════════════════════════════════════════════════════════════════


def load_npz(path: Path):
    """Load a predictions .npz and return (gt, pred) as CPU LongTensors."""
    data = np.load(path)
    #gt   = torch.from_numpy(data["y_gt"].astype("int64"))    # (N, T, 6)
    #pred = torch.from_numpy(data["y_pred"].astype("int64"))  # (N, T, 6)
    gt = torch.from_numpy(data["gt"].astype("int64"))  # (N, T, 6)
    pred = torch.from_numpy(data["pred"].astype("int64"))  # (N, T, 6)

    return gt, pred

def main():
    for model_name in MODEL_NAMEs:
        model_path = _MODELS_ROOT / model_name
        out_dir    = model_path
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'═' * 70}")
        print(f"Model: {model_name}")
        print(f"{'═' * 70}")

        for split in SPLITS:
            cache_path = out_dir / f"predictions.npz"

            if not cache_path.exists():
                print(f"  [{split}] No cache found at {cache_path} – skipping.")
                continue

            gt, pred = load_npz(cache_path)
            m = tab_metrics(gt, pred)

            predicted_item, target_item = vectors_to_text_token(pred, gt)
            output_path = model_path / "predictions.txt"

            print_results(target_item, predicted_item, output_path)

            out_path = out_dir / f"metrics_{split}.txt"
            print_tab_metrics(m, save_path=str(out_path), prefix=f"{model_name} | {split}")
            print(f"  [{split}] Metrics saved → {out_path}")


if __name__ == "__main__":
    main()
