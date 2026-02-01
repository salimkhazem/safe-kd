from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from data.corruptions import CORRUPTIONS, apply_corruption
from data.transforms import build_transforms
from eval.metrics import accuracy, collect_logits
from eval.early_exit import safe_risk_curve
from methods.safe_exit import calibrate_thresholds


class CIFAR10CDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


class SyntheticCorruptionDataset(Dataset):
    def __init__(self, base, corruption: str, severity: int, transform=None):
        self.base = base
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        np.random.seed(idx + self.severity * 10000)
        img = apply_corruption(img, self.corruption, self.severity)
        if self.transform:
            img = self.transform(img)
        return img, label


def _load_cifar10c_np(data_dir: str, corruption: str, severity: int):
    data_dir = Path(data_dir)
    img_path = data_dir / f"{corruption}.npy"
    lbl_path = data_dir / "labels.npy"
    if not img_path.exists() or not lbl_path.exists():
        return None
    images = np.load(img_path)
    labels = np.load(lbl_path)
    start = (severity - 1) * 10000
    end = start + 10000
    return images[start:end], labels[start:end]


def evaluate_cifar10c(cfg, model, device, thresholds=None) -> Dict[str, float]:
    results: Dict[str, float] = {}
    risk_accumulator = {}
    transform = build_transforms(cfg.data.input_size, False, cfg.data.augment)
    base = datasets.CIFAR10(root=cfg.data.data_dir, train=False, download=True)
    for corruption in CORRUPTIONS:
        for severity in [1, 3, 5]:
            loaded = _load_cifar10c_np(cfg.robustness.cifar10c_dir, corruption, severity)
            if loaded is not None:
                images, labels = loaded
                ds = CIFAR10CDataset(images, labels, transform=transform)
            else:
                ds = SyntheticCorruptionDataset(base, corruption, severity, transform=transform)
            loader = DataLoader(ds, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=cfg.data.num_workers)
            logits, labels = collect_logits(model, loader, device)
            acc = accuracy(logits[-1], labels)
            results[f"{corruption}_s{severity}"] = acc
            if thresholds is not None:
                risks = safe_risk_curve(logits, labels, thresholds)
                for delta, stats in risks.items():
                    risk_accumulator.setdefault(delta, []).append(stats["overall_risk"])
    results["mean_corruption_acc"] = float(np.mean(list(results.values()))) if results else 0.0
    if risk_accumulator:
        for delta, vals in risk_accumulator.items():
            results[f"risk_delta_{delta}"] = float(np.mean(vals))
    return results
