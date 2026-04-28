from math import pi
from torch import Tensor
import torch.nn as nn
from typing import Tuple
from Embeddigns import NumberEmbedder
from DiffusionUtils import UniformDistribution, extend_dim, LinearSchedule
import torch
from tqdm import tqdm
from einops import repeat
import torch.nn.functional as F
from AuxiliaryLoss import pc_tokens_to_binary, fret_distance, cof_chord_distance, jaccard_tonal_distance, hand_span_penalty, string_activity_jaccard_loss

N_STRINGS  = 6
N_CLASSES  = 24   # 0=muted, 1=open, 2..20=fret1..19
PAD_FRET = -1
PAD_PC = -1
OPEN_PITCHES = [40, 45, 50, 55, 59, 64]

class DiffusionModel(nn.Module):
    """Simple diffusion model implementation."""

    def __init__(self, model, noise_steps=100, embed_dim=32,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.model = model.to(device)
        self.noise_steps = noise_steps
        self.embed_dim = embed_dim
        self.device = device
        self.encoder = NumberEmbedder(128, device=device)
        self.max_grad_norm = 1.0  # Prevent exploding gradients
        self.diffusion_dim = N_STRINGS * self.embed_dim   # 6*32, the flat diffusion space

        self.embeddings = nn.Embedding(N_CLASSES, self.embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T, 6, n_classes) float one-hot  OR  (B, T, 6) integer IDs
        out : (B, T, 6*embed_dim)  continuous embedding ready for diffusion
        """
        if x.ndim == 4:
            ids = x.argmax(dim=-1)          # (B, T, 6)  integer class IDs
        else:
            ids = x.long()                  # already integer IDs

        emb = self.embeddings(ids)           # (B, T, 6, embed_dim)
        B, T = emb.shape[:2]
        return emb.view(B, T, N_STRINGS * self.embed_dim)  # (B, T, 6*embed_dim)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, T, 6*embed_dim)  denoised latent
        out  : (B, T, 6, n_classes) logits — argmax(-1) gives class IDs
        """
        B, T = x.shape[:2]
        x_str = x.view(B, T, N_STRINGS, self.embed_dim)    # (B, T, 6, E)
        W = self.embeddings.weight                           # (n_classes, E)
        logits = x_str @ W.T                               # (B, T, 6, n_classes)
        return logits


    def prepare_noise_schedule(self):
        """Linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def noise_audios(self, x, sigmas_batch):
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        return x_noisy, noise, v_target

    def sample_timesteps(self, batch_size, device, dim):
        """Sample random timesteps."""
        sigmas = UniformDistribution()(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch
        # return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))

    def sample_timesteps_val(self, batch_size, device, dim):
        """Sample random timesteps."""
        # sigmas = UniformDistribution()(num_samples=batch_size, device=device)
        sigmas = torch.Tensor([0.2, 0.4, 0.6, 0.8]).to(device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _rounding_loss(self, x0_pred: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        x0_pred    : (B, T, 6*embed_dim)
        target_ids : (B, T, 6)  integer class IDs
        """
        logits = self.decode(x0_pred)                      # (B, T, 6, n_classes)
        # CE expects (N, C) and (N,)
        return F.cross_entropy(
            logits.reshape(-1, N_CLASSES),             # (B*T*6, n_classes)
            target_ids.reshape(-1)                          # (B*T*6,)
        )

    def avg_acc(self, x0_pred: torch.Tensor, target_ids: torch.Tensor) -> float:
        logits = self.decode(x0_pred)                      # (B, T, 6, n_classes)
        pred_cls = logits.argmax(dim=-1)                   # (B, T, 6)
        return (pred_cls == target_ids).float().mean().item()

    def tab_loss(self, pred, target):
        # pred   : (B, T, 126) raw logits from model
        # target : (B, T, 6, 22) one-hot
        B, T = target.shape[:2]
        logits = pred.view(B, T, N_STRINGS, N_CLASSES)  # (B, T, 6, 22)
        target = target.view(B, T, N_STRINGS, N_CLASSES)  # (B, T, 6, 22)
        target_idx = target.argmax(dim=-1)  # (B, T, 6)
        # cross-entropy expects (B, C, ...) layout
        loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),  # (B, 22, T, 6)
            target_idx,  # (B, T, 6)
        )
        # loss = F.cross_entropy(
        #     logits.reshape(-1, N_CLASSES),  # (B*T*6, 21)
        #     target_idx.reshape(-1),  # (B*T*6,)
        # )
        return loss

    @staticmethod
    def _build_pc_frets(tab: torch.Tensor):
        """
        Derive frets and pitch-class vectors from a (6, 21) one-hot tab matrix.

        Returns
        -------
        frets : torch.Tensor, shape (6,)   int64   -1=muted, 0=open, 1..19=fret
        pc    : torch.Tensor, shape (6,)   int64   -1=muted, 0..11=pitch class
        """
        frets = torch.full((tab.shape[0], N_STRINGS,), PAD_FRET, dtype=torch.long)
        pc = torch.full((tab.shape[0], N_STRINGS,), PAD_PC, dtype=torch.long)

        classes = tab
        if len(tab.shape) == 4:
            classes = tab.argmax(dim=-1)  # (6,)  — class index per string
        for t in range(tab.shape[0]):
            for s in range(N_STRINGS):
                cls = int(classes[t, s].item())
                if cls == 0:
                    # muted — keep PAD values
                    continue
                fret = cls - 1  # class 1 → fret 0 (open), class 2 → fret 1, …
                frets[t, s] = fret
                pc[t, s] = (OPEN_PITCHES[s] + fret) % 12

        return frets, pc

    def _forward_pass(self, target, prev, audio, cond, sigmas_fn):
        # Save integer IDs BEFORE embedding (needed for rounding loss)
        target_ids = target.argmax(dim=-1)  # (B, T, 6)

        target_emb = self.encode(target)  # (B, T, 6*E)
        prev_emb = self.encode(prev)  # (B, T, 6*E)

        sigmas_t, sigmas_batch_t = sigmas_fn(
            target_emb.shape[0], self.device, target_emb.ndim
        )

        x_t, _, v_target = self.noise_audios(target_emb, sigmas_batch_t)

        sigmas_t_enc = self.encoder.to_embedding(sigmas_t)
        predicted_v = self.model(x_t, sigmas_t_enc, prev_emb, audio, cond)

        alphas_b, betas_b = self.get_alpha_beta(sigmas_batch_t)
        x0_pred = alphas_b * x_t - betas_b * predicted_v  # (B, T, 6*E)

        embed_loss = F.mse_loss(x0_pred, target_emb)  # anchor in emb space
        rounding_loss = self._rounding_loss(x0_pred, target_ids)

        loss = embed_loss + 0.1 * rounding_loss
        return loss, x0_pred, target_ids

    def train_step(self, optimizer, batch, losses_str):
        optimizer.zero_grad()
        target, prev, audio, cond = batch

        loss, x0_pred, target_ids = self._forward_pass(
            target, prev, audio, cond, self.sample_timesteps
        )

        x0 = torch.argmax(self.decode(x0_pred), dim=-1)
        pc_preds, frets_preds, frets, pcs = [], [], [], []
        for b in range(x0_pred.size(0)):
            frets_pred, pc_pred = self._build_pc_frets(x0[b])
            fret, pc = self._build_pc_frets(target_ids[b])
            frets.append(fret)
            pcs.append(pc)
            frets_preds.append(frets_pred)
            pc_preds.append(pc_pred)

        frets_preds = torch.stack(frets_preds, dim=0).to(self.device)
        pc_preds = torch.stack(pc_preds, dim=0).to(self.device)
        frets = torch.stack(frets, dim=0).to(self.device)
        pcs = torch.stack(pcs, dim=0).to(self.device)

        # (B, 6) raw pc indices → (B, 12) binary chord vectors
        pc_gt_bin = pc_tokens_to_binary(pcs.to(self.device))  # ground truth
        pc_pred_bin = pc_tokens_to_binary(pc_preds.to(self.device))  # from argmax prediction
        hs_loss = hand_span_penalty(pc_pred_bin).mean()
        string_loss = string_activity_jaccard_loss(frets_preds, frets).mean()
        fret_loss = fret_distance(frets_preds, frets).mean()
        pc_loss = jaccard_tonal_distance(pc_pred_bin, pc_gt_bin).mean()
        cof_loss = cof_chord_distance(pc_pred_bin, pc_gt_bin).mean()
        #cof_loss = cof_chord_distance(pc_vec_a=pc_pred_bin, pc_vec_b=pc_gt_bin, fret_a=frets_preds, fret_b=frets).mean()

        if "f" in losses_str:
            loss = loss + 0.1 * fret_loss
        if "p" in losses_str:
            loss = loss + pc_loss
        if "c" in losses_str:
            loss = loss + cof_loss
        if "s" in losses_str:
            loss = loss + string_loss
        if "h" in losses_str:
            loss = loss + hs_loss
        #loss = loss + 0.1 * fret_loss + pc_loss + cof_loss  # + string_loss + hs_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss.item()

    def val_step(self, batch):
        target, prev, audio, cond = batch

        loss, x0_pred, target_ids = self._forward_pass(
            target, prev, audio, cond, self.sample_timesteps_val
        )

        acc = self.avg_acc(x0_pred, target_ids)
        return loss.item(), acc

    @torch.no_grad()
    def sample(self, input, prev_input, audio, cond, num_steps):
        """Sample new audios from the diffusion model."""
        input = self.encode(input)  # (B, seq_len, embed_dim)
        if prev_input is not None:
            prev_input = self.encode(prev_input)  # (B, seq_len, embed_dim)
        x_noisy = torch.randn_like(input).to(self.device)
        x_noisy /= x_noisy.max()
        b = x_noisy.shape[0]
        sigmas = LinearSchedule()(num_steps + 1, device=x_noisy.device)
        sigmas_batch = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas_batch, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        sigmas_t_encoded = self.encoder.to_embedding(sigmas)
        sigmas_t_encoded_batch = repeat(sigmas_t_encoded, "l i -> l b i", b=b)
        # sigmas_t_encoded_batch = extend_dim(sigmas_t_encoded_batch, dim=x_noisy.ndim + 1)

        progress_bar = tqdm(range(num_steps), disable=True)
        # Progressively denoise the audios
        # for i in tqdm(reversed(range(1, num_steps)), desc="Sampling...", total=self.noise_steps - 1):
        for i in progress_bar:
            v_pred = self.model(x_noisy, sigmas_t_encoded_batch[i], prev_input, audio, cond)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i + 1]:.2f})")
            # if return_process:
            #   intermediate_audios.append(x_noisy.cpu())
        return x_noisy
