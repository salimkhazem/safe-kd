from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def ece_score(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    labels = labels.to(preds.device)
    acc = preds.eq(labels).float()

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.any():
            ece += torch.abs(acc[mask].mean() - confs[mask].mean()) * mask.float().mean()
    return float(ece.item())


def reliability_bins(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> Dict[str, np.ndarray]:
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    acc = preds.eq(labels).float()

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_acc = []
    bin_conf = []
    bin_counts = []
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.any():
            bin_acc.append(acc[mask].mean().item())
            bin_conf.append(confs[mask].mean().item())
            bin_counts.append(mask.float().mean().item())
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_counts.append(0.0)
    return {
        "bin_acc": list(bin_acc),
        "bin_conf": list(bin_conf),
        "bin_counts": list(bin_counts),
    }
