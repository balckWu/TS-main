from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch


def make_balanced_domain_schedule(domains: List[str], total_iters: int) -> List[str]:
    """Create a roughly domain-balanced episode order for one epoch."""
    if len(domains) == 0:
        raise ValueError("domains cannot be empty")
    reps, rem = divmod(int(total_iters), len(domains))
    schedule: List[str] = []
    for _ in range(reps):
        chunk = list(domains)
        random.shuffle(chunk)
        schedule.extend(chunk)
    if rem > 0:
        tail = list(domains)
        random.shuffle(tail)
        schedule.extend(tail[:rem])
    return schedule


class EMATaskMemoryBank:
    """Per-domain EMA prototype bank for task tokens.

    The project no longer uses support retrieval.  This memory bank stores a
    stable task-token prototype for each domain and is used directly during
    validation/inference.
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = float(momentum)
        self.bank: Dict[str, torch.Tensor] = {}

    def has(self, class_id: str) -> bool:
        return class_id in self.bank

    def get(self, class_id: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if class_id not in self.bank:
            return None
        value = self.bank[class_id]
        return value.to(device) if device is not None else value

    @torch.no_grad()
    def update(self, class_id: str, task_tokens: torch.Tensor, importance: Optional[torch.Tensor] = None) -> None:
        if task_tokens.ndim != 3:
            raise ValueError(f"task_tokens must have shape [B, T, C], got {tuple(task_tokens.shape)}")
        if importance is None:
            weights = torch.full(
                (task_tokens.shape[0],),
                1.0 / max(task_tokens.shape[0], 1),
                device=task_tokens.device,
                dtype=task_tokens.dtype,
            )
        else:
            weights = importance.to(device=task_tokens.device, dtype=task_tokens.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)

        prototype = (task_tokens.detach() * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True).cpu()
        if class_id in self.bank:
            self.bank[class_id] = self.momentum * self.bank[class_id] + (1.0 - self.momentum) * prototype
        else:
            self.bank[class_id] = prototype

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().cpu() for k, v in self.bank.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.bank = {str(k): v.clone().cpu() for k, v in state_dict.items()}

    def summary(self) -> str:
        if not self.bank:
            return "empty"
        return ", ".join(f"{k}: {tuple(v.shape)}" for k, v in self.bank.items())
