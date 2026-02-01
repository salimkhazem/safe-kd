from __future__ import annotations

from typing import List, Optional

import torch

from methods.dkd import dkd_loss
from methods.kd import kd_loss
from methods.losses import cross_entropy_loss


def _normalize_weights(weights: List[float], n: int) -> List[float]:
    if weights is None or len(weights) != n:
        return [1.0 / n for _ in range(n)]
    s = sum(weights)
    if s <= 0:
        return [1.0 / n for _ in range(n)]
    return [w / s for w in weights]


def compute_loss(
    logits: List[torch.Tensor],
    targets: torch.Tensor,
    method_cfg,
    teacher_logits: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    name = method_cfg.name
    num_exits = len(logits)
    weights = _normalize_weights(list(method_cfg.exit_weights), num_exits)

    if name == "erm":
        return cross_entropy_loss(logits[-1], targets)

    total = 0.0
    deep_logits = logits[-1].detach()
    teacher_final = teacher_logits[-1] if teacher_logits is not None else None

    for j, (logit_j, w_j) in enumerate(zip(logits, weights)):
        ce = cross_entropy_loss(logit_j, targets)
        loss = ce
        if name in {"kd", "dkd", "safe_kd"} and teacher_final is not None:
            if name == "kd":
                loss = loss + method_cfg.alpha * kd_loss(logit_j, teacher_final, method_cfg.temperature)
            else:
                loss = loss + method_cfg.alpha * dkd_loss(
                    logit_j, teacher_final, targets, temperature=method_cfg.temperature, alpha=1.0, beta=1.0
                )
        if name == "safe_kd" and j < num_exits - 1:
            loss = loss + method_cfg.beta * kd_loss(logit_j, deep_logits, method_cfg.temperature)
        total = total + w_j * loss

    return total
