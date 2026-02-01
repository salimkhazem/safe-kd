from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from methods.calibration import ece_score


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels).item())


def ece(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return ece_score(logits, labels)


def collect_logits(model, loader, device) -> Tuple[List[torch.Tensor], torch.Tensor]:
    model.eval()
    all_logits: List[List[torch.Tensor]] = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits_list = model(x)
            if not all_logits:
                all_logits = [[] for _ in logits_list]
            for j, logits in enumerate(logits_list):
                all_logits[j].append(logits.cpu())
            all_labels.append(y.cpu())
    all_logits = [torch.cat(xs, dim=0) for xs in all_logits]
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels
