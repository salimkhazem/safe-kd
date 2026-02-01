from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from methods.safe_exit import naive_exit_predictions, safe_exit_predictions
from models.flops import expected_compute


def _stats_from_preds(preds: torch.Tensor, exits: torch.Tensor, labels: torch.Tensor, num_exits: int):
    stats = {"exit_rates": [], "exit_acc": [], "exit_risk": []}
    for j in range(num_exits):
        mask = exits == j
        rate = float(mask.float().mean().item())
        if mask.any():
            acc = float((preds[mask] == labels[mask]).float().mean().item())
            risk = 1.0 - acc
        else:
            acc = 0.0
            risk = 0.0
        stats["exit_rates"].append(rate)
        stats["exit_acc"].append(acc)
        stats["exit_risk"].append(risk)
    stats["expected_compute"] = expected_compute(stats["exit_rates"])
    overall_acc = sum(r * a for r, a in zip(stats["exit_rates"], stats["exit_acc"]))
    stats["overall_acc"] = overall_acc
    stats["overall_risk"] = 1.0 - overall_acc
    return stats


def safe_risk_curve(logits_list: List[torch.Tensor], labels: torch.Tensor, thresholds: Dict[float, List[float]]):
    curves = {}
    for delta, tau in thresholds.items():
        out = safe_exit_predictions(logits_list, tau)
        stats = _stats_from_preds(out["preds"], out["exits"], labels, len(logits_list))
        curves[str(delta)] = stats
    return curves


def naive_risk_curve(logits_list: List[torch.Tensor], labels: torch.Tensor, thresholds: List[float]):
    curves = {}
    for t in thresholds:
        out = naive_exit_predictions(logits_list, t)
        stats = _stats_from_preds(out["preds"], out["exits"], labels, len(logits_list))
        curves[str(t)] = stats
    return curves
