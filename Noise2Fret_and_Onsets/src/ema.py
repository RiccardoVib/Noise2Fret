"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999,
                 warmup: bool = True, update_every: int = 1,
                 device=None):
        """
        decay        : target decay. 0.999 averages over roughly the last
                       1/(1-decay) = 1000 updates.
        warmup       : ramp the decay in from 0 over the first updates, so
                       the shadow tracks the model quickly at the start
                       instead of being anchored to random init.
        update_every : update the shadow every N optimizer steps. Leave at 1
                       unless the copy shows up in your profile.
        """
        self.decay = decay
        self.warmup = warmup
        self.update_every = update_every
        self.num_updates = 0
        self.step = 0

        self.shadow = {}
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point:
                self.shadow[name] = p.detach().clone().to(device or p.device)

        self.buffers = {}
        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone().to(device or b.device)

        self._backup = None

    def _current_decay(self):
        if not self.warmup:
            return self.decay
        # standard warmup schedule: slow at first, approaching self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Call once after each optimizer.step()."""
        self.step += 1
        if self.step % self.update_every != 0:
            return

        d = self._current_decay()
        self.num_updates += 1

        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            if p.requires_grad:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                # frozen params (e.g. the U-Net during the count-head phase)
                self.shadow[name].copy_(p.detach())

        for name, b in model.named_buffers():
            if name in self.buffers:
                self.buffers[name].copy_(b.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Overwrite the model's weights with the EMA weights, saving the
        originals so restore() can put them back."""
        self._backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name in self.shadow
        }
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is None:
            return
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name].data)
        self._backup = None

    @contextmanager
    def average_parameters(self, model: nn.Module):
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "step": self.step,
            "shadow": self.shadow,
            "buffers": self.buffers,
        }

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.warmup = sd["warmup"]
        self.update_every = sd.get("update_every", 1)
        self.num_updates = sd["num_updates"]
        self.step = sd.get("step", sd["num_updates"])
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}
        self.buffers = {k: v.clone() for k, v in sd["buffers"].items()}

    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        self.buffers = {k: v.to(device) for k, v in self.buffers.items()}
        return self


@torch.no_grad()
def ema_model_copy(model: nn.Module, ema: EMA) -> nn.Module:
    """A standalone deep copy carrying the EMA weights — useful for saving a
    clean inference checkpoint without disturbing the live model."""
    out = copy.deepcopy(model)
    for name, p in out.named_parameters():
        if name in ema.shadow:
            p.data.copy_(ema.shadow[name].data)
    for name, b in out.named_buffers():
        if name in ema.buffers:
            b.data.copy_(ema.buffers[name].data)
    return out
