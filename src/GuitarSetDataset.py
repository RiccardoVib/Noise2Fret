# 1. Compute full CQT  →  (n_frames, 192)
# 2. Convert JAMS annotations to frame indices  →  (n_frames, 6, 21)
# 3. For each frame k: slice CQT[k-4:k+5]  →  store as repr[k]  shape (9, 192)
# 4. Transpose  →  (192, 9)
# 5. labels[k] = tab_matrix[k]
# 6. Save both arrays together in .npz


import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from utils import find_folder_upward

current_dir = Path(os.getcwd())
print(f"current_dir: {current_dir}")
files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)
npz_dir = files_dir / "GuitarSet/c/"


class TabDataset(Dataset):
    def __init__(self, npz_dir):

        all_repr = []
        all_labels = []

        for fname in sorted(os.listdir(npz_dir)):
            if fname.endswith(".npz"):
                data = np.load(os.path.join(npz_dir, fname))
                all_repr.append(data["repr"])  # (n_frames, 192, 9)
                all_labels.append(data["labels"])  # (n_frames, 6, 21)

        # Concatenate all tracks into one flat dataset
        repr_all = np.concatenate(all_repr, axis=0)  # (N_total_frames, 192, 9)
        labels_all = np.concatenate(all_labels, axis=0)  # (N_total_frames, 6, 21)

        self.X = torch.tensor(repr_all,   dtype=torch.float32)   # (N, 192, 9)
        self.Y = torch.tensor(labels_all, dtype=torch.float32)   # (N, 6, 21)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
        # X[idx]: (192, 9)   ← audio input
        # Y[idx]: (6, 21)    ← tab target for that single frame