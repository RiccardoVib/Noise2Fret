import torch
from torch.utils.data import Dataset
import numpy as np
import os

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
