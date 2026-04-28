"""
ComputeMetrics.py – load cached .npz predictions and compute metrics.
No model / dataset / GPU required.
Just edit the CONFIG block and run.
"""

from pathlib import Path
import numpy as np
import torch

from tab_metrics import tab_metrics, print_tab_metrics
from diffusion_trainingEMB import vectors_to_text_token, print_results
# ══════════════════════════════════════════════════════════════════════════════
# CONFIG ← edit this block only
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ["test", "Validation"]

_MODELS_ROOT = _SCRIPT_DIR.parent.parent.parent / "TrainedModels" / "ToConsider"
MODEL_NAMEs = [
    "Audio2Tab_Unet_H_64_E_32_I_515_U_True_feat_alls__TabEmbLoss_01_moredeep_drop01"
]

#
# _MODELS_ROOT = _SCRIPT_DIR.parent.parent.parent / "TrainedModels" / "ToConsider" / "Ablations"
# MODEL_NAMEs = [
#     "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_513_U_True_feat_stft_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_514_U_True_feat_stft+sf_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_514_U_True_feat_stft+b_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_1_U_True_feat_sf_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_1_U_True_feat_b_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_2_U_True_feat_sf+b_TabEmb01",
# ]
SPLITS = ["test"]

# _MODELS_ROOT = _SCRIPT_DIR.parent.parent.parent / "TrainedModels" / "ToConsider" / "Loss"
# MODEL_NAMEs = [
    #"Audio2Tab_Unet_H_64_I_515_U_True_feat_all_c_TabEmbLoss_01",
    #           "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_f_TabEmbLoss_01",
    #           "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_p_TabEmbLoss_01",
    #           "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_h_TabEmbLoss_01",
    #           "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_s_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_fc_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_fh_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_fp_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_fpc_TabEmbLoss_01",
    #            "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_fpcs_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_fs_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_pc_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_ps_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_s_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_sh_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_ch_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_ph_TabEmbLoss_01",
    #             "Audio2Tab_Unet_H_64_I_515_U_True_feat_all_cs_TabEmbLoss_01",
    #            ]

# SPLITS = ["test"]

# # Root folder where model checkpoints live
# _MODELS_ROOT = _SCRIPT_DIR.parent.parent.parent / "TrainedModels" / "ToConsider" / "HiddenDims"
#
# MODEL_NAMEs = [
#     "Audio2Tab_Unet_H_16_I_515_U_True_feat_all_TabEmb01",
#     "Audio2Tab_Unet_H_32_I_515_U_True_feat_all_TabEmb01",
#     "Audio2Tab_Unet_H_64_I_515_U_True_TabEmb01",
#     "Audio2Tab_Unet_H_128_I_515_U_True_TabEmb01",
# ]
# SPLITS = ["test"]

# MODEL_NAMEs = ["Audio2Tab_B_128TabCNN_GOAT"]
# _MODELS_ROOT = _SCRIPT_DIR.parent.parent.parent / "TrainedModels" / "ToConsider" / "ComparisonsGOAT"
# SPLITS = ["test"]

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
