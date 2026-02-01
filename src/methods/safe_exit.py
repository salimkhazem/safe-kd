from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    scores = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - delta))) - 1
    k = min(max(k, 0), n - 1)
    return float(scores[k])


def calibrate_thresholds(
    logits_list: List[torch.Tensor], labels: torch.Tensor, deltas: List[float]
) -> Dict[float, List[float]]:
    thresholds = {float(d): [] for d in deltas}
    labels_np = labels.cpu().numpy()
    for logits in logits_list:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        scores = 1.0 - probs[np.arange(len(labels_np)), labels_np]
        for d in deltas:
            thresholds[float(d)].append(conformal_quantile(scores, d))
    return thresholds


def safe_exit_predictions(
    logits_list: List[torch.Tensor], thresholds: List[float]
) -> Dict[str, torch.Tensor]:
    num_exits = len(logits_list)
    probs_list = [torch.softmax(l, dim=1) for l in logits_list]
    preds = []
    exits = []
    for i in range(probs_list[0].shape[0]):
        exit_idx = num_exits - 1
        for j in range(num_exits):
            max_prob = probs_list[j][i].max().item()
            if max_prob >= 1.0 - thresholds[j]:
                exit_idx = j
                break
        exits.append(exit_idx)
        preds.append(probs_list[exit_idx][i].argmax().item())
    return {"preds": torch.tensor(preds), "exits": torch.tensor(exits)}


def naive_exit_predictions(
    logits_list: List[torch.Tensor], threshold: float
) -> Dict[str, torch.Tensor]:
    num_exits = len(logits_list)
    probs_list = [torch.softmax(l, dim=1) for l in logits_list]
    preds = []
    exits = []
    for i in range(probs_list[0].shape[0]):
        exit_idx = num_exits - 1
        for j in range(num_exits):
            max_prob = probs_list[j][i].max().item()
            if max_prob >= threshold:
                exit_idx = j
                break
        exits.append(exit_idx)
        preds.append(probs_list[exit_idx][i].argmax().item())
    return {"preds": torch.tensor(preds), "exits": torch.tensor(exits)}
