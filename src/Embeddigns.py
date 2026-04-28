import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from torch import Tensor
from typing import Union, Sequence
from math import pi


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256, device='cpu'):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2, device=device))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features).to(device)

        # self.to(device)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class RandomFourierEmbedding(nn.Module):
    """
    Random Fourier Features (RFF) embedding for continuous inputs.

    This is particularly useful for:
    - Time embeddings in diffusion models
    - Positional encodings
    - Function approximation with neural networks

    The embedding maps scalar inputs to high-dimensional feature vectors
    using random Fourier features: [cos(2πBx), sin(2πBx)]
    where B is a random matrix sampled from a Gaussian distribution.
    """

    def __init__(self, input_dim=1, embedding_dim=256, scale=1.0, learnable=False):
        """
        Args:
            input_dim (int): Dimension of input (usually 1 for time/scalar)
            embedding_dim (int): Dimension of output embedding (should be even)
            scale (float): Scale parameter for the Gaussian distribution
            learnable (bool): Whether the random frequencies are learnable parameters
        """
        super().__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be even"

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.scale = scale

        # Random frequencies matrix B ~ N(0, scale²I)
        B = torch.randn(embedding_dim // 2, input_dim) * scale

        if learnable:
            self.register_parameter('B', nn.Parameter(B))
        else:
            self.register_buffer('B', B)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        # Ensure x has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension

        # Compute 2πBx
        projections = 2 * np.pi * torch.mm(x.view(-1, self.input_dim), self.B.t())

        # Compute [cos(2πBx), sin(2πBx)]
        cos_proj = torch.cos(projections)
        sin_proj = torch.sin(projections)

        # Concatenate cos and sin components
        embeddings = torch.cat([cos_proj, sin_proj], dim=-1)

        # Reshape back to original batch dimensions
        original_shape = x.shape[:-1]
        return embeddings.view(*original_shape, self.embedding_dim)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Projection layer used in many diffusion models.
    This is a specific type of Random Fourier Features.
    """

    def __init__(self, embedding_dim=256, scale=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Time steps, shape (batch_size,) or (batch_size, 1)
        """
        if x.dim() > 1:
            x = x.squeeze(-1)

        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as used in Transformers.
    This is a deterministic version of Fourier embeddings.
    """

    def __init__(self, embedding_dim=256, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             -(np.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Position indices, shape (batch_size,) or (batch_size, seq_len)
        """
        if x.dim() == 1:
            return self.pe[x.long()]
        else:
            return self.pe[x.long()]


class TimeEmbedding(nn.Module):
    """
    Complete time embedding module for diffusion models.
    Combines Random Fourier Features with MLPs.
    """

    def __init__(self, embedding_dim=256, hidden_dim=512, scale=16.0):
        super().__init__()

        self.fourier_proj = GaussianFourierProjection(embedding_dim, scale)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        """
        Args:
            t: Time steps, shape (batch_size,)

        Returns:
            Time embeddings, shape (batch_size, hidden_dim)
        """
        t_emb = self.fourier_proj(t)
        return self.mlp(t_emb)


def visualize_embeddings():
    """Visualize different types of embeddings."""

    # Test inputs
    x = torch.linspace(0, 1, 100)

    # Initialize different embedding types
    rff = RandomFourierEmbedding(input_dim=1, embedding_dim=64, scale=10.0)
    gaussian_proj = GaussianFourierProjection(embedding_dim=64, scale=16.0)
    sinusoidal = SinusoidalPositionalEmbedding(embedding_dim=64)

    # Generate embeddings
    rff_emb = rff(x)
    gaussian_emb = gaussian_proj(x)
    sinusoidal_emb = sinusoidal(torch.arange(100))

    # Plot embeddings
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Random Fourier Features
    axes[0, 0].imshow(rff_emb.T.detach().numpy(), aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Random Fourier Features')
    axes[0, 0].set_xlabel('Input Position')
    axes[0, 0].set_ylabel('Embedding Dimension')

    # Show some individual components
    axes[1, 0].plot(x.numpy(), rff_emb[:, :5].detach().numpy())
    axes[1, 0].set_title('RFF: First 5 Components')
    axes[1, 0].set_xlabel('Input Value')
    axes[1, 0].set_ylabel('Embedding Value')

    # Gaussian Fourier Projection
    axes[0, 1].imshow(gaussian_emb.T.detach().numpy(), aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Gaussian Fourier Projection')
    axes[0, 1].set_xlabel('Input Position')
    axes[0, 1].set_ylabel('Embedding Dimension')

    axes[1, 1].plot(x.numpy(), gaussian_emb[:, :5].detach().numpy())
    axes[1, 1].set_title('GFP: First 5 Components')
    axes[1, 1].set_xlabel('Input Value')
    axes[1, 1].set_ylabel('Embedding Value')

    # Sinusoidal Positional Embedding
    axes[0, 2].imshow(sinusoidal_emb.T.detach().numpy(), aspect='auto', cmap='viridis')
    axes[0, 2].set_title('Sinusoidal Positional Embedding')
    axes[0, 2].set_xlabel('Position Index')
    axes[0, 2].set_ylabel('Embedding Dimension')

    axes[1, 2].plot(sinusoidal_emb[:, :5].detach().numpy())
    axes[1, 2].set_title('Sinusoidal: First 5 Components')
    axes[1, 2].set_xlabel('Position Index')
    axes[1, 2].set_ylabel('Embedding Value')

    plt.tight_layout()
    plt.show()


def test_embeddings():
    """Test the embedding modules."""

    print("Testing Random Fourier Embeddings...")

    # Test Random Fourier Features
    rff = RandomFourierEmbedding(input_dim=1, embedding_dim=128, scale=5.0)
    x = torch.randn(32, 1)  # Batch of 32 scalar inputs
    rff_output = rff(x)
    print(f"RFF Input shape: {x.shape}, Output shape: {rff_output.shape}")

    # Test with different input shapes
    x_2d = torch.randn(16, 10, 1)  # Sequence input
    rff_output_2d = rff(x_2d)
    print(f"RFF 2D Input shape: {x_2d.shape}, Output shape: {rff_output_2d.shape}")

    # Test Gaussian Fourier Projection
    gfp = GaussianFourierProjection(embedding_dim=256, scale=16.0)
    t = torch.randn(32)  # Time steps
    gfp_output = gfp(t)
    print(f"GFP Input shape: {t.shape}, Output shape: {gfp_output.shape}")

    # Test Time Embedding
    time_emb = TimeEmbedding(embedding_dim=128, hidden_dim=256)
    time_output = time_emb(t)
    print(f"Time Embedding Input shape: {t.shape}, Output shape: {time_output.shape}")

    # Test Sinusoidal Positional Embedding
    pos_emb = SinusoidalPositionalEmbedding(embedding_dim=512, max_len=1000)
    positions = torch.arange(50)  # Position indices
    pos_output = pos_emb(positions)
    print(f"Positional Embedding Input shape: {positions.shape}, Output shape: {pos_output.shape}")

    print("\nAll tests passed!")


class ImprovedUNetWithRFF(nn.Module):
    """
    Example of integrating Random Fourier Features into your U-Net.
    This shows how to use proper time embeddings in diffusion models.
    """

    def __init__(self, in_channels=1, out_channels=1, hidden_dims=64, time_emb_dim=256):
        super().__init__()

        # Time embedding
        self.time_embedding = TimeEmbedding(
            embedding_dim=128,
            hidden_dim=time_emb_dim,
            scale=16.0
        )

        # Downsampling path
        self.conv1 = nn.Conv1d(in_channels + 1, hidden_dims, kernel_size=3, padding=1)  # +1 for conditioning
        self.conv2 = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=3, padding=1)
        self.down1 = nn.MaxPool1d(2)

        # Time projection layers
        self.time_proj1 = nn.Linear(time_emb_dim, hidden_dims)
        self.time_proj2 = nn.Linear(time_emb_dim, hidden_dims * 2)
        self.time_proj3 = nn.Linear(time_emb_dim, hidden_dims * 4)

        self.conv3 = nn.Conv1d(hidden_dims, hidden_dims * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_dims * 2, hidden_dims * 2, kernel_size=3, padding=1)
        self.down2 = nn.MaxPool1d(2)

        # Bottleneck
        self.conv5 = nn.Conv1d(hidden_dims * 2, hidden_dims * 4, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(hidden_dims * 4, hidden_dims * 4, kernel_size=3, padding=1)

        # Upsampling path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv1d(hidden_dims * 4 + hidden_dims * 2, hidden_dims * 2, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(hidden_dims * 2, hidden_dims * 2, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = nn.Conv1d(hidden_dims * 2 + hidden_dims, hidden_dims, kernel_size=3, padding=1)
        self.conv10 = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=3, padding=1)

        self.out = nn.Conv1d(hidden_dims, out_channels, kernel_size=1)

    def forward(self, x, t, c):
        """
        Args:
            x: Input audio, shape (batch, channels, length)
            t: Time steps, shape (batch,)
            c: Conditioning, shape (batch, 1, length)
        """
        # Get time embeddings
        t_emb = self.time_embedding(t)  # (batch, time_emb_dim)

        # Concatenate input with conditioning
        x = torch.cat([x, c], dim=1)

        # Downsampling with time injection
        x1 = F.relu(self.conv1(x))
        x1 = x1 + self.time_proj1(t_emb)[:, :, None]  # Broadcast time embedding
        x1 = F.relu(self.conv2(x1))
        x2 = self.down1(x1)

        x2 = F.relu(self.conv3(x2))
        x2 = x2 + self.time_proj2(t_emb)[:, :, None]
        x2 = F.relu(self.conv4(x2))
        x3 = self.down2(x2)

        # Bottleneck
        x3 = F.relu(self.conv5(x3))
        x3 = x3 + self.time_proj3(t_emb)[:, :, None]
        x3 = F.relu(self.conv6(x3))

        # Upsampling with skip connections
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        return self.out(x)


if __name__ == "__main__":
    # Run tests
    test_embeddings()

    # Visualize embeddings
    visualize_embeddings()

    # Test improved U-Net
    print("\nTesting Improved U-Net with RFF...")
    model = ImprovedUNetWithRFF(in_channels=1, out_channels=1, hidden_dims=64)

    batch_size = 4
    audio_length = 1024
    x = torch.randn(batch_size, 1, audio_length)
    t = torch.randint(0, 1000, (batch_size,)).float()
    c = torch.randn(batch_size, 1, audio_length)

    output = model(x, t, c)
    print(f"U-Net Input: {x.shape}, Time: {t.shape}, Conditioning: {c.shape}")
    print(f"U-Net Output: {output.shape}")
    print("Success!")
