"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import math
from Embeddings import NumberEmbedder
import torch
import torch.nn as nn
import torch.nn.functional as F


def audio_positional_encoding(length, channels, device, dtype):
    """Sinusoidal PE over the audio time axis.
    """
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(channels // 2, device=device, dtype=dtype).unsqueeze(0)
    rate = 1.0 / torch.pow(10000.0, (2 * i) / channels)
    ang = pos * rate
    pe = torch.zeros(length, channels, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.T.unsqueeze(0)  # (1, C, L)


class AudioCrossAttention(nn.Module):
    """Event positions (queries) attend over audio frames (keys/values).
    """

    TIME_BIAS_MODES = ("none", "hard_times", "soft_cumulative")

    def __init__(self, channels, audio_ch, num_heads=4, dropout=0.1,
                 use_audio_pe=True, max_events=16,
                 time_bias_mode="soft_cumulative", init_sigma=0.08, init_sigma_idx=0.7,
                 init_sharpen=0.25
                 ):

        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.max_events = max_events
        self.query_pe = nn.Embedding(max_events, channels)
        nn.init.normal_(self.query_pe.weight, std=0.02)

        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.use_audio_pe = use_audio_pe
        assert time_bias_mode in self.TIME_BIAS_MODES, time_bias_mode
        self.time_bias_mode = time_bias_mode

        self.norm_q = ChannelLayerNorm(channels)
        self.norm_kv = ChannelLayerNorm(audio_ch)

        self.to_q = nn.Conv1d(channels, channels, 1)
        self.to_kv = nn.Conv1d(audio_ch, channels * 2, 1)
        self.to_out = nn.Conv1d(channels, channels, 1)

        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        if time_bias_mode != "none":
            self.log_sigma = nn.Parameter(torch.full((num_heads,), math.log(init_sigma)))
            self.log_sigma_idx = nn.Parameter(torch.full((num_heads,), math.log(init_sigma_idx)))
            self.bias_gate = nn.Parameter(torch.ones(num_heads))

            self.onset_sharpen = nn.Parameter(torch.full((num_heads,), init_sharpen))


    def _soft_cumulative_bias(self, prob, n_events, q_valid, L_q, L_k, n_slots, device, dtype):
        """Bias built from the onset POSTERIORGRAM, without peak picking.

        Normalized cumulative onset mass is an "expected event index" for every
        audio frame:

            c(j)   = sum_{k<=j} p(k) / sum_k p(k)     in [0, 1]
            idx(j) = c(j) * n_events                  in [0, n_events]

        Event slot i should then attend where idx(j) ~ i + 0.5, and the bias is
        Gaussian in that index distance.

        """
        B = prob.shape[0]
        p = prob.to(device=device, dtype=dtype).clamp(min=0.0).unsqueeze(1)
        if p.shape[-1] != L_k:
            p = F.interpolate(p, size=L_k, mode="linear", align_corners=False).clamp(min=0.0)
        p = p.squeeze(1)  # (B, L_k)

        c = p.cumsum(dim=-1)
        total = c[:, -1:].clamp(min=1e-6)
        c_mid = (c - 0.5 * p) / total  # midpoint rule
        n = n_events.to(device=device, dtype=dtype).clamp(min=1.0).unsqueeze(-1)
        idx_k = (c_mid * n).unsqueeze(1).unsqueeze(2)  # (B,1,1,L_k)

        step = n_slots / max(L_q, 1)
        idx_q = ((torch.arange(L_q, device=device, dtype=dtype) + 0.5) * step).view(1, 1, L_q, 1)

        sigma = self.log_sigma_idx.exp().clamp(min=1e-2).view(1, -1, 1, 1)
        gate = self.bias_gate.view(1, -1, 1, 1)
        bias = (-gate * (idx_k - idx_q) ** 2 / (2 * sigma ** 2)).clamp(min=-30.0)

        log_p = torch.log(p.clamp(min=1e-3)).view(B, 1, 1, L_k)  # floor at -6.9
        w = self.onset_sharpen.view(1, -1, 1, 1)
        bias = bias + (w * log_p).clamp(-15.0, 15.0)

        v_q = q_valid.to(device=device, dtype=dtype).unsqueeze(1)
        if v_q.shape[-1] != L_q:
            v_q = F.adaptive_max_pool1d(v_q, L_q)
        return bias * v_q.squeeze(1).view(B, 1, L_q, 1)

    def _time_bias(self, q_times, q_valid, L_q, L_k, device, dtype):
        """Gaussian log-bias on |t_query - t_key|, from peak-picked slot times.
        """
        B = q_times.shape[0]
        t_q = q_times.to(device=device, dtype=dtype).unsqueeze(1)
        v_q = q_valid.to(device=device, dtype=dtype).unsqueeze(1)

        if t_q.shape[-1] != L_q:
            t_q = F.interpolate(t_q, size=L_q, mode="linear", align_corners=False)
            v_q = F.adaptive_max_pool1d(v_q, L_q)

        t_k = torch.linspace(0.0, 1.0, L_k, device=device, dtype=dtype)
        d = (t_q.squeeze(1).unsqueeze(-1) - t_k.view(1, 1, L_k)).unsqueeze(1)

        sigma = self.log_sigma.exp().clamp(min=1e-3).view(1, -1, 1, 1)
        gate = self.bias_gate.view(1, -1, 1, 1)
        bias = (-gate * d ** 2 / (2 * sigma ** 2)).clamp(min=-30.0)

        return bias * v_q.squeeze(1).view(B, 1, L_q, 1)

    def forward(self, x, audio_feat, mask=None, return_attn=False, onset_cond=None):
        """
        x          : (B, C, L_q)  event-position features
        audio_feat : (B, A, L_k)  audio frames, NOT resampled to L_q
        mask       : (B, 1, L_q)  1.0 real event, 0.0 padding, or None
        onset_cond : dict from the onset head, or None to disable the bias
            "prob"     : (B, L_frames) onset posteriorgram in [0, 1]
            "n_events" : (B,) how many events the window contains
            "times"    : (B, L_slots) peak-picked onset time per slot in [0, 1]
            "valid"    : (B, L_slots) which slots carry a real onset
            "n_slots"  : int, event-slot resolution the above are given at
        """
        residual = x
        B, C, L_q = x.shape

        h = self.norm_q(x)
        qpos = self.query_pe(torch.arange(L_q, device=x.device))  # (L_q, C)
        h = h + qpos.T.unsqueeze(0)  # (1, C, L_q)

        a = self.norm_kv(audio_feat)
        if self.use_audio_pe:
            a = a + audio_positional_encoding(a.shape[-1], a.shape[1], a.device, a.dtype)

        q = self.to_q(h)
        k, v = self.to_kv(a).chunk(2, dim=1)
        L_k = k.shape[-1]

        H, d = self.num_heads, C // self.num_heads
        q = q.view(B, H, d, L_q).transpose(-2, -1)  # (B, H, L_q, d)
        k = k.view(B, H, d, L_k).transpose(-2, -1)  # (B, H, L_k, d)
        v = v.view(B, H, d, L_k).transpose(-2, -1)  # (B, H, L_k, d)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,L_q,L_k)

        if self.time_bias_mode != "none" and onset_cond is not None:
            times, prob = onset_cond.get("times"), onset_cond.get("prob")
            valid = onset_cond.get("valid")
            n_slots = onset_cond.get("n_slots") or (
                valid.shape[-1] if valid is not None else
                times.shape[-1] if times is not None else L_q)
            if valid is None:
                valid = torch.ones(B, n_slots, device=attn.device, dtype=attn.dtype)

            if self.time_bias_mode == "soft_cumulative" and prob is not None:
                attn = attn + self._soft_cumulative_bias(
                    prob, onset_cond["n_events"], valid, L_q, L_k, n_slots,
                    attn.device, attn.dtype)
            elif times is not None:
                attn = attn + self._time_bias(times, valid, L_q, L_k,
                                              attn.device, attn.dtype)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L_q, d)
        out = out.transpose(-2, -1).reshape(B, C, L_q)
        out = self.to_out(out)

        if mask is not None:
            m = mask if mask.shape[-1] == L_q else F.adaptive_max_pool1d(mask.float(), L_q)
            out = out * m

        out = residual + out

        if return_attn:
            # (B, H, L_q, L_k)
            return out, attn
        return out


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
    """Feature modulation (FiLM)"""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


def downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resize a mask to a different temporal resolution.

    """
    if mask.dtype != torch.float32:
        mask = mask.float()
    if mask.shape[-1] == target_len:
        return mask

    return F.adaptive_max_pool1d(mask, target_len)


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if mask is None:
        return x
    return x * downsample_mask(mask, x.shape[-1])


class MaskedSelfAttention(nn.Module):

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, C, L = x.shape
        residual = x
        h = self.norm(x)

        qkv = self.qkv(h).view(B, 3, C, L)
        q, k, v = qkv.unbind(1)

        # attn[b, i, j] = q_i . k_j   (softmax over j = keys)
        attn = torch.matmul(q.transpose(-2, -1), k) / (C ** 0.5)  # (B, L, L)

        if mask is not None:
            m = downsample_mask(mask, L)  # (B, 1, L)
            attn = attn.masked_fill(m.bool().logical_not(), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = self.proj(out)
        out = self.dropout(out)

        return residual + out


class FeedForward(nn.Module):
    """Simple feedforward layer"""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),  #
            nn.Conv1d(channels * 2, channels, 1)
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class InjectionBlock(nn.Module):
    """
    Feature injection with:
    1) local projection
    2) FiLM conditioning
    3) gated residual injection
    """

    def __init__(self, channels, inject_channels):
        super().__init__()
        # self.inject_proj = nn.Conv1d(inject_channels, channels, 1)
        self.inject_proj = nn.Sequential(
            nn.Conv1d(inject_channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.film = nn.Conv1d(inject_channels, channels * 2, kernel_size=1)
        self.gate = nn.Conv1d(inject_channels, channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, inject_feat):
        if inject_feat is None:
            return x
        # Match length
        if inject_feat.shape[-1] != x.shape[-1]:
            if inject_feat.shape[-1] > x.shape[-1]:
                inject_feat = F.adaptive_avg_pool1d(inject_feat, x.shape[-1])  # downsample: average
            else:
                inject_feat = F.interpolate(inject_feat, size=x.shape[-1], mode='linear')

        local = self.inject_proj(inject_feat)  # [B, C, L]
        gamma, beta = self.film(inject_feat).chunk(2, dim=1)  # [B, C, L], [B, C, L]
        gate = torch.tanh(self.gate(inject_feat))  # [B, C, L]

        h = x * (1.0 + gamma) + beta
        h = h + gate * local

        return h


class OnsetConditioning(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(3, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, onset_cond, L, device, dtype):
        if onset_cond is None:
            return None
        prob = onset_cond.get("prob")
        if prob is None:
            return None

        p = prob.to(device=device, dtype=dtype).clamp(min=0.0).unsqueeze(1)  # (B,1,L_frames)
        B = p.shape[0]
        p_L = F.interpolate(p, size=L, mode="linear", align_corners=False).clamp(min=0.0)  # (B,1,L)

        c = p.cumsum(dim=-1)
        total = c[:, :, -1:].clamp(min=1e-6)
        c_mid = (c - 0.5 * p) / total  # midpoint rule, in [0, 1]
        n_events = onset_cond.get("n_events")
        n = (n_events.to(device=device, dtype=dtype).clamp(min=1.0).view(B, 1, 1)
             if n_events is not None else
             torch.ones(B, 1, 1, device=device, dtype=dtype))
        idx = (c_mid * n) / n  # normalize back to [0,1] so channel scale is stable across n_events
        idx_L = F.interpolate(idx, size=L, mode="linear", align_corners=False)

        valid = onset_cond.get("valid")
        if valid is not None:
            v = valid.to(device=device, dtype=dtype).unsqueeze(1)
            v_L = v if v.shape[-1] == L else F.adaptive_max_pool1d(v, L)
        else:
            v_L = torch.ones(B, 1, L, device=device, dtype=dtype)

        feat = torch.cat([p_L, idx_L, v_L], dim=1)  # (B, 3, L)
        return self.proj(feat)



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


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each time position."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x, mask=None):  # (B, C, L)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class MaskConditioning(nn.Module):
    """Turn the (B, 1, L) mask into a feature the U-Net can consume, so the
    denoiser *knows* which positions are structural rather than inferring it.
    """

    def __init__(self, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(2, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, mask):  # (B, 1, L)

        B, _, L = mask.shape
        pos = torch.arange(L, device=mask.device, dtype=mask.dtype)
        pos = (pos / max(L - 1, 1)).view(1, 1, L).expand(B, 1, L)
        return self.proj(torch.cat([mask, pos], dim=1))


class ResNetBlock(nn.Module):
    """ResNet block with all components"""

    def __init__(self, in_ch, out_ch, time_dim, use_attention=False, audio_ch=None, inject_ch=None, dropout=0.1,
                 zero_pad_output=True, use_cross_attn=False,
                 time_bias_mode="soft_cumulative", max_events=16):
        super().__init__()

        self.zero_pad_output = zero_pad_output

        self.norm1 = ChannelLayerNorm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        # --- block2 ---
        self.norm2 = ChannelLayerNorm(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.drop2 = nn.Dropout(dropout)

        # Residual connection
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Components
        self.modulation = Modulation(out_ch, time_dim)
        self.attention = MaskedSelfAttention(out_ch, dropout=dropout) if use_attention else None
        self.feedforward = FeedForward(out_ch, dropout=dropout)
        self.injection = InjectionBlock(out_ch, inject_ch) if inject_ch else None

        self.audio_cross = (
            AudioCrossAttention(out_ch, audio_ch, num_heads=4, dropout=dropout,
                                max_events=max_events, time_bias_mode=time_bias_mode)
            if (audio_ch and use_cross_attn) else None
        )

        self.onset_cond_proj = OnsetConditioning(out_ch) if not use_cross_attn else None
        self.onset_inject = InjectionBlock(out_ch, out_ch) if not use_cross_attn else None

    def _norm(self, norm_layer, x, mask):
        return norm_layer(x, mask)

    def forward(self, x, time_emb, audio_feat=None, inject_feat=None, mask=None,
                onset_cond=None):
        # x: (B, seq_len, in_ch)
        # time_emb: (B, time_dim)
        # Main path
        h = self.conv1(self.act1(self._norm(self.norm1, x, mask)))
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.drop2(self.conv2(self.act2(self._norm(self.norm2, h, mask))))

        # Add residual
        h = h + self.shortcut(x)

        # Apply components
        h = self.modulation(h, time_emb)

        if inject_feat is not None:
            h = self.injection(h, inject_feat)

        if self.audio_cross is not None and audio_feat is not None:  # <-- new
            h = self.audio_cross(h, audio_feat, mask, onset_cond=onset_cond)
        elif self.onset_inject is not None:  # <-- ablation: onset-only, no cross-attn
            onset_feat = self.onset_cond_proj(onset_cond, h.shape[-1], h.device, h.dtype)
            h = self.onset_inject(h, onset_feat)

        if self.attention:
            h = self.attention(h, mask)
        h = self.feedforward(h)
        if self.zero_pad_output:
            h = apply_mask(h, mask)

        return h


class TokenUNet(nn.Module):
    def __init__(
            self,
            in_channels,
            base_channels=64,
            time_dim=128,
            max_len=64,
            inject_feature_dim=1,
            audio_embed_dim=64,
            dropout=0.1,
            zero_pad_output=True,
            time_bias_mode="soft_cumulative",
            use_cross_attn=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_bias_mode = time_bias_mode
        self.pos_emb = nn.Embedding(max_len, in_channels)
        self.time_encoder = NumberEmbedder(time_dim)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_dropout = nn.Dropout(dropout)

        # Input
        self.input_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        self.c_conv = nn.Conv1d(inject_feature_dim, base_channels, 3, padding=1)
        self.audio_encoder = AudioEncoder(audio_embed_dim)
        self.mask_cond = MaskConditioning(base_channels)

        # Encoder
        self.down1 = ResNetBlock(base_channels, base_channels * 2, time_dim, use_attention=True,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                                 zero_pad_output=zero_pad_output)
        self.down2 = ResNetBlock(base_channels * 2, base_channels * 4, time_dim, use_attention=True,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                                 zero_pad_output=zero_pad_output)
        self.down3 = ResNetBlock(base_channels * 4, base_channels * 8, time_dim, use_attention=True,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                                 zero_pad_output=zero_pad_output, use_cross_attn=use_cross_attn,
                                 time_bias_mode=time_bias_mode, max_events=max_len)
        self.down4 = ResNetBlock(base_channels * 8, base_channels * 8, time_dim, use_attention=True,
                                 audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                                 zero_pad_output=zero_pad_output, use_cross_attn=use_cross_attn,
                                 time_bias_mode=time_bias_mode, max_events=max_len)

        self.downsample = nn.Conv1d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        # Middle
        self.mid = ResNetBlock(base_channels * 8, base_channels * 8, time_dim, use_attention=True,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                               zero_pad_output=zero_pad_output, use_cross_attn=use_cross_attn,
                               time_bias_mode=time_bias_mode, max_events=max_len)

        # Decoder
        self.upsample = nn.ConvTranspose1d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1)

        self.up1 = ResNetBlock(base_channels * 16, base_channels * 4, time_dim, use_attention=True,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                               zero_pad_output=zero_pad_output, use_cross_attn=use_cross_attn,
                               time_bias_mode=time_bias_mode, max_events=max_len)
        self.up2 = ResNetBlock(base_channels * 12, base_channels * 2, time_dim, use_attention=True,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                               zero_pad_output=zero_pad_output)
        self.up3 = ResNetBlock(base_channels * 6, base_channels, time_dim, use_attention=True,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                               zero_pad_output=zero_pad_output)
        self.up4 = ResNetBlock(base_channels * 3, base_channels, time_dim, use_attention=True,
                               audio_ch=audio_embed_dim, inject_ch=base_channels, dropout=dropout,
                               zero_pad_output=zero_pad_output)

        # Output
        self.output = nn.Sequential(
            ChannelLayerNorm(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, noisy_tokens, sigmas_t, inject_audio, inject_features, mask,
                onset_cond=None):
        """
        Args:
            noisy_tokens: (B, seq_len, E) - token indices
            time: (B,) - diffusion timestep
            inject_features: (B, seq_len, inject_channels) - optional conditioning features

        Returns:
            logits: (B, seq_len, vocab_size) - predicted token logits
        """
        B, L, E = noisy_tokens.shape

        sigmas_t_enc = self.time_encoder.to_embedding(sigmas_t)

        # Time embedding
        t = self.time_emb(sigmas_t_enc)

        self.pos_ids = torch.arange(L, device=noisy_tokens.device).unsqueeze(0)  # [1, T]
        pos = self.pos_emb(self.pos_ids).permute(0, 2, 1)

        x = noisy_tokens.permute(0, 2, 1) + pos

        x = self.input_dropout(self.input_conv(x))
        freq_features = self.c_conv(inject_features.permute(0, 2, 1))
        if mask is not None:
            x = x + self.mask_cond(mask)  # MaskConditioning

        audio_feat = self.audio_encoder(inject_audio) if inject_audio is not None else None

        oc = dict(onset_cond=onset_cond)

        # Encoder with skip connections
        skip1 = self.down1(x, t, audio_feat, freq_features, mask=None, **oc)
        skip2 = self.down2(skip1, t, audio_feat, freq_features, mask=None, **oc)
        skip3 = self.down3(skip2, t, audio_feat, freq_features, mask=None, **oc)
        skip4 = self.down4(skip3, t, audio_feat, freq_features, mask=None, **oc)

        # Downsample
        x = self.downsample(skip4)
        mask_mid = downsample_mask(mask, x.shape[-1]) if mask is not None else None
        mask_mid = None
        # Middle
        x = self.mid(x, t, audio_feat, freq_features, mask=mask_mid, **oc)

        # Upsample
        x = self.upsample(x)

        # Decoder with skip connections
        if x.shape[-1] != skip4.shape[-1]:
            x = F.interpolate(x, size=skip4.shape[-1], mode='linear', align_corners=False)

        x = self.up1(torch.cat([x, skip4], dim=1), t, audio_feat, freq_features, mask=None, **oc)
        x = self.up2(torch.cat([x, skip3], dim=1), t, audio_feat, freq_features, mask=None, **oc)
        x = self.up3(torch.cat([x, skip2], dim=1), t, audio_feat, freq_features, mask=None, **oc)
        x = self.up4(torch.cat([x, skip1], dim=1), t, audio_feat, freq_features, mask=None, **oc)

        # Output logits
        return self.output(x).permute(0, 2, 1)  # (B, seq_len, vocab_size)