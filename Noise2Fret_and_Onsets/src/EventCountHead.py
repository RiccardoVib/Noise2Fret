"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import torch
import torch.nn as nn
from U_NET_Token_Masked import AudioEncoder, ChannelLayerNorm

class TemporalBlock(nn.Module):
    """Local temporal processing before pooling."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm1 = ChannelLayerNorm(channels)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm2 = ChannelLayerNorm(channels)
        self.act2 = nn.SiLU()

    def forward(self, x):  # x: (B, C, L)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.act2(self.norm2(self.conv2(h)))
        return x + h


class AttentionPool1d(nn.Module):
    """Learned weighted pooling over the time axis instead of a plain mean.
    Lets the head focus on the frames that actually carry onset/count
    information rather than averaging them away."""

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):  # x: (B, C, L)
        attn_logits = self.score(x)          # (B, 1, L)
        attn = torch.softmax(attn_logits, dim=-1)
        pooled = (x * attn).sum(dim=-1)       # (B, C)
        return pooled, attn.squeeze(1)        # (B, C), (B, L)


class EventCountHead(nn.Module):

    def __init__(self, audio_ch, spectral_ch, hidden=128, max_events=13, dropout=0.1):
        super().__init__()

        self.audio_ch = audio_ch
        self.spectral_ch = spectral_ch
        self.hidden = hidden

        self.audio_block = TemporalBlock(audio_ch, dropout)
        self.spectral_block = TemporalBlock(spectral_ch, dropout)

        self.audio_pool = AttentionPool1d(audio_ch)
        self.spectral_pool = AttentionPool1d(spectral_ch)

        self.audio_mean_pool = nn.AdaptiveAvgPool1d(1)
        self.spectral_mean_pool = nn.AdaptiveAvgPool1d(1)

        in_dim = 2 * (audio_ch + spectral_ch)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, max_events),  # logits over 0..max_events-1
        )
        self.c_conv = nn.Conv1d(514, hidden, 3, padding=1)
        self.audio_encoder = AudioEncoder(audio_ch)

    def forward(self, audio, cond, return_attn=False):

        audio_feat = self.audio_encoder(audio)
        freq_features = self.c_conv(cond.permute(0, 2, 1))

        a = self.audio_block(audio_feat)
        f = self.spectral_block(freq_features)

        a_attn_pooled, a_attn = self.audio_pool(a)
        f_attn_pooled, f_attn = self.spectral_pool(f)

        a_mean_pooled = self.audio_mean_pool(a).squeeze(-1)
        f_mean_pooled = self.spectral_mean_pool(f).squeeze(-1)

        feat = torch.cat(
            [a_attn_pooled, f_attn_pooled, a_mean_pooled, f_mean_pooled], dim=-1
        )
        logits = self.mlp(feat)

        if return_attn:
            return logits, (a_attn, f_attn)
        return logits
