import librosa
import numpy as np
import torch

import torch.nn.functional as F
from einops import rearrange

def compute_audio_features(audio: torch.Tensor, sr: int = 16000,
                            n_fft: int = 1024, hop_length: int = 256) -> dict:
    """
    audio: (B, L, 1)
    returns dict with STFT magnitude, spectral flux, and brightness per batch item
    """
    # (B, L, 1) -> (B, L) -> (B, 1, L) for stft input
    x = audio.squeeze(-1)  # (B, L)

    window = torch.hann_window(n_fft).to(x.device)

    # --- STFT: output shape (B, F, T) ---
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)  # (B, F, T)
    mag = stft.abs()  # (B, F, T)

    # ── CQT ──────────────────────────────────────────────────────────

    fmin = librosa.note_to_hz('E2')
    audio_float = x.cpu().numpy()
    audio_padded = np.pad(audio_float, ((0, 0), (0, 2496)), mode='reflect')  # (128, 2048)

    cqt = torch.tensor(
            librosa.cqt(
                audio_padded,
                sr=16000,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=96,
                bins_per_octave=24,
                pad_mode='reflect',  # default is 'constant' (zero-pad)
                filter_scale=0.5
            )
        )

    cqt_mag = cqt.abs().permute(0, 2, 1).to(audio.device)  # (n_frames, n_bins)

    n_frames = len(cqt_mag)

    # --- Spectral Flux: L1 difference between consecutive frames ---
    flux = torch.diff(mag, dim=-1)           # (B, F, T-1)
    flux = flux.clamp(min=0).sum(dim=1)      # half-wave rectify + sum over freqs -> (B, T-1)
    #flux = flux.clamp(min=0).pow(2).sum(dim=1).sqrt()
    flux = F.pad(flux, (1, 0), value=0.0)  # (B, T)

    # --- Spectral Brightness: energy above cutoff_freq / total energy ---
    freqs = torch.linspace(0, sr / 2, mag.shape[1], device=x.device)  # (F,)
    cutoff_hz = 1500.0
    bright_mask = freqs >= cutoff_hz          # (F,) boolean
    energy_total = mag.pow(2).sum(dim=1)      # (B, T)
    energy_bright = mag[:, bright_mask, :].pow(2).sum(dim=1)  # (B, T)
    brightness = (energy_bright / energy_total.clamp(min=1e-8)) # (B, T)

    mag = rearrange(mag, 'b f t -> b t f')

    return {
        "cqt_mag": cqt_mag[:, :7],          # (B, T, F)
        "stft_mag": mag,          # (B, T, F)
        "spectral_flux": flux.unsqueeze(-1),   # (B, T)
        "brightness": brightness.unsqueeze(-1),     # (B, T)
    }
