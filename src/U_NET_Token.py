import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)  # [seq_len, 1]
    i = torch.arange(d_model // 2).unsqueeze(0)  # [1, d_model/2]

    angle_rates = 1 / torch.pow(10000, (2 * i) / d_model)
    angle = pos * angle_rates  # [seq_len, d_model/2]

    # Interleave sin and cos
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class Modulation(nn.Module):
    """Simple feature modulation (FiLM)"""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


class SelfAttention(nn.Module):
    """Simple self-attention with dropout"""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).view(B, 3, C, L)
        q, k, v = qkv.unbind(1)

        # Simple attention with dropout
        attn = torch.matmul(q.transpose(-2, -1), k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # Dropout on attention weights
        out = torch.matmul(v, attn.transpose(-2, -1))

        out = self.proj(out)
        out = self.dropout(out)  # Dropout on output

        return residual + out


class FeedForward(nn.Module):
    """Simple feedforward layer"""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.ff = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),  #
            nn.Conv1d(channels * 2, channels, 1)
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class InjectionBlock(nn.Module):
    """Simple feature injection"""

    def __init__(self, channels, inject_channels):
        super().__init__()
        self.proj = nn.Conv1d(inject_channels, channels, 1)

    def forward(self, x, inject_feat):
        if inject_feat is None:
            return x
        # Match length
        if inject_feat.shape[-1] != x.shape[-1]:
            inject_feat = F.interpolate(inject_feat, size=x.shape[-1], mode='linear')
        return x + self.proj(inject_feat)


class AudioEncoder(nn.Module):
    """Compress raw waveform B×1×T → B×audio_embed_dim×L_compressed"""

    def __init__(self, audio_embed_dim=64, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),  # T/4
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),  # T/16
            nn.SiLU(),
            nn.Conv1d(32, audio_embed_dim, kernel_size=5, stride=4, padding=2),  # T/64
            nn.SiLU(),
            nn.Dropout(p=dropout)  #
        )
        # 16000 / (4*4*4) = 250 → usable spatial embedding

    def forward(self, audio):  # audio: B×16000×1
        return self.encoder(audio.permute(0, 2, 1))  # → B×audio_embed_dim×250


class ResNetBlock(nn.Module):
    """Simple ResNet block with all components"""

    def __init__(self, in_ch, out_ch, time_dim, use_attention=False, audio_ch=None, inject_ch=None, dropout=0.1):
        super().__init__()

        # Main conv blocks
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1)
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.Dropout(dropout)  #
        )

        # Residual connection
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Components
        self.modulation = Modulation(out_ch, time_dim)
        self.attention = SelfAttention(out_ch) if use_attention else None
        self.feedforward = FeedForward(out_ch, dropout=dropout)
        self.injection = InjectionBlock(out_ch, inject_ch) if inject_ch else None
        if audio_ch:
            self.audio_injection = InjectionBlock(out_ch, audio_ch) if audio_ch else None

        self.dropout = nn.Dropout(dropout)  # pass dropout param here

    def forward(self, x, time_emb, audio_feat=None, inject_feat=None):
        # x: (B, seq_len, in_ch)
        # time_emb: (B, time_dim)
        # Main path
        h = self.block1(x)
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.block2(h)

        # Add residual
        h = h + self.shortcut(x)

        # Apply components
        h = self.modulation(h, time_emb)

        if inject_feat is not None:
            h = self.injection(h, inject_feat)
        if audio_feat is not None:
            h = self.audio_injection(h, audio_feat)
        if self.attention:
            h = self.attention(h)

        h = self.feedforward(h)

        return h


class TokenUNet(nn.Module):
    """U-Net adapted for token-level prediction"""

    def __init__(
            self,
            in_channels,
            base_channels=64,
            time_dim=128,
            max_len=64,
            inject_feature_dim=1,
            audio_embed_dim=64,
            dropout=0.1,
            use_pre=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pos_emb = nn.Embedding(max_len, in_channels)
        self.use_pre = use_pre

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_dropout = nn.Dropout(dropout)

        token_in = in_channels * 2 if use_pre else in_channels

        # Input
        self.input_conv = nn.Conv1d(token_in, base_channels, 3, padding=1)
        self.c_conv = nn.Conv1d(inject_feature_dim, base_channels, 3, padding=1)
        self.audio_encoder = AudioEncoder(audio_embed_dim)

        # Encoder
        self.down1 = ResNetBlock(base_channels, base_channels * 2, time_dim, audio_ch=audio_embed_dim,
                                 inject_ch=base_channels, dropout=dropout)
        self.down2 = ResNetBlock(base_channels * 2, base_channels * 4, time_dim, use_attention=False,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout)
        self.down3 = ResNetBlock(base_channels * 4, base_channels * 8, time_dim, use_attention=False,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout)

        self.down4 = ResNetBlock(base_channels * 8, base_channels * 8, time_dim, use_attention=False,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout)

        self.downsample = nn.Conv1d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        # Middle
        self.mid = ResNetBlock(base_channels * 8, base_channels * 8, time_dim, use_attention=True,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout)

        # Decoder
        self.upsample = nn.ConvTranspose1d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1)

        self.up1 = ResNetBlock(base_channels * 16, base_channels * 4, time_dim, use_attention=False,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout)
        self.up2 = ResNetBlock(base_channels * 12, base_channels * 2, time_dim, use_attention=False,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout)
        self.up3 = ResNetBlock(base_channels * 6, base_channels, time_dim, audio_ch=audio_embed_dim,
                               inject_ch=base_channels, dropout=dropout)

        self.up4 = ResNetBlock(base_channels * 3, base_channels, time_dim, audio_ch=audio_embed_dim,
                               inject_ch=base_channels, dropout=dropout)

        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, noisy_tokens, time, pre_frames, inject_audio=None, inject_features=None):
        """
        Args:
            noisy_tokens: (B, seq_len, E) - token indices
            time: (B,) - diffusion timestep
            inject_features: (B, seq_len, inject_channels) - optional conditioning features

        Returns:
            logits: (B, seq_len, vocab_size) - predicted token logits
        """
        B, L, E = noisy_tokens.shape

        # Time embedding
        t = self.time_emb(time)

        self.pos_ids = torch.arange(L, device=noisy_tokens.device).unsqueeze(0)  # [1, T]
        pos = self.pos_emb(self.pos_ids).permute(0, 2, 1)

        noisy_tokens = noisy_tokens.permute(0, 2, 1) + pos

        # Input
        if self.use_pre:
            pre_frames = pre_frames.permute(0, 2, 1) + pos
            x = torch.cat([pre_frames, noisy_tokens], dim=1)
        else:
            x = noisy_tokens

        x = self.input_dropout(self.input_conv(x))  #
        # inject_features = F.pad(inject_features, (1, 0))
        freq_features = self.c_conv(inject_features.permute(0, 2, 1))
        audio_feat = self.audio_encoder(inject_audio) if inject_audio is not None else None

        # Encoder with skip connections
        skip1 = self.down1(x, t, audio_feat, freq_features)
        skip2 = self.down2(skip1, t, audio_feat, freq_features)
        skip3 = self.down3(skip2, t, audio_feat, freq_features)
        skip4 = self.down4(skip3, t, audio_feat, freq_features)

        # Downsample
        x = self.downsample(skip4)

        # Middle
        x = self.mid(x, t, audio_feat, freq_features)

        # Upsample
        x = self.upsample(x)

        # Decoder with skip connections
        if x.shape[-1] != skip4.shape[-1]:
            x = F.interpolate(x, size=skip4.shape[-1], mode='linear', align_corners=False)
        x = self.up1(torch.cat([x, skip4], dim=1), t, audio_feat, freq_features)
        x = self.up2(torch.cat([x, skip3], dim=1), t, audio_feat, freq_features)
        x = self.up3(torch.cat([x, skip2], dim=1), t, audio_feat, freq_features)
        x = self.up4(torch.cat([x, skip1], dim=1), t, audio_feat, freq_features)

        # Output logits
        return self.output(x).permute(0, 2, 1)  # (B, seq_len, vocab_size)