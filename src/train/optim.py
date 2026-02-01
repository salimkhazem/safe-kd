from typing import Tuple

import torch
from torch.optim import AdamW


def build_optimizer(model, lr: float, weight_decay: float):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / max(1, warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
