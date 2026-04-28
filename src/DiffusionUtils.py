import json
import torch
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn as nn
from typing import Any, Tuple

def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

# Function to save losses to file
def save_losses(train_losses, val_losses, filename='losses.json'):
    losses_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(filename, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")


# Function to load losses from file
def load_losses(filename='losses.json'):
    with open(filename, 'r') as f:
        losses_dict = json.load(f)
    return losses_dict['train_losses'], losses_dict['val_losses']


# Function to plot losses
def plot_losses(train_losses, val_losses, filename='loss_plot.png'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Over Time', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)

class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin

def check_rqvae_reconstruction(diffusion_model, test_ids: torch.Tensor, verbose=True):
    """
    Test that the RQVAE round-trips correctly: token IDs → latent → token IDs.

    Args:
        diffusion_model: your DiffusionModel instance (rqvae must be loaded)
        test_ids: LongTensor of shape (B, seq) with integer chord IDs
    """
    dm = diffusion_model
    device = dm.device
    x = test_ids.to(device)  # (B, seq)

    # --- Encode: token IDs → quantized latent ---
    latent = dm._encode(x)  # (B, seq, H)

    # --- Decode: quantized latent → logits → argmax ---
    chosen_ids, _ = dm._decode(latent)  # (B, seq)

    # --- Metrics ---
    correct = (chosen_ids == x).float()
    token_acc = correct.mean().item()
    seq_acc = correct.all(dim=-1).float().mean().item()  # full sequence must match

    if verbose:
        print(f"Input IDs:        {x[0].tolist()}")
        print(f"Reconstructed:    {chosen_ids[0].tolist()}")
        print(f"Token accuracy:   {token_acc * 100:.2f}%")
        print(f"Sequence accuracy:{seq_acc * 100:.2f}%  (all tokens correct)")
        mismatches = (chosen_ids != x).nonzero(as_tuple=False)
        if mismatches.numel() == 0:
            print("✅ Perfect reconstruction!")
        else:
            print(f"❌ {mismatches.shape[0]} mismatch(es) at positions: {mismatches.tolist()}")

    return token_acc, seq_acc, chosen_ids
