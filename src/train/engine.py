from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from eval.metrics import accuracy, collect_logits, ece, nll
from methods.losses import mixup_data, mixup_criterion
from methods.training_objectives import compute_loss
from train.ema import ModelEMA
from train.optim import build_optimizer, build_scheduler
from train.seed import set_seed
from utils.io import ensure_dir


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, teacher=None):
    model.train()
    total_loss = 0.0
    num = 0
    use_mixup = cfg.data.augment.mixup
    amp_enabled = bool(cfg.train.amp and torch.cuda.is_available())
    accum_steps = max(1, int(cfg.train.grad_accum_steps))
    show_progress = bool(getattr(cfg.train, "show_progress", True))
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for step, (x, y) in enumerate(
        tqdm(loader, desc="train", leave=False, disable=not show_progress, dynamic_ncols=True), start=1
    ):
        x = x.to(device)
        y = y.to(device)
        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y)
        with autocast(enabled=amp_enabled):
            logits_list = model(x)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher.ema(x)
            else:
                if cfg.method.name in {"kd", "dkd", "safe_kd"}:
                    teacher_logits = [l.detach() for l in logits_list]
                else:
                    teacher_logits = None
            if use_mixup:
                loss = 0.0
                for logits in logits_list:
                    loss = loss + mixup_criterion(logits, y_a, y_b, lam)
                loss = loss / len(logits_list)
            else:
                loss = compute_loss(logits_list, y, cfg.method, teacher_logits)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None:
                teacher.update(model)
        total_loss += loss.item() * accum_steps * x.size(0)
        num += x.size(0)
    if step > 0 and step % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if teacher is not None:
            teacher.update(model)
    return total_loss / max(1, num)


def evaluate_metrics(model, loader, device) -> Dict[str, float]:
    logits_list, labels = collect_logits(model, loader, device)
    metrics: Dict[str, float] = {}
    for j, logits in enumerate(logits_list):
        metrics[f"acc_exit{j+1}"] = accuracy(logits, labels)
        metrics[f"nll_exit{j+1}"] = nll(logits, labels)
        metrics[f"ece_exit{j+1}"] = ece(logits, labels)
    metrics["acc_final"] = metrics.get(f"acc_exit{len(logits_list)}", 0.0)
    return metrics


def train_model(model, loaders, cfg, run_dir: str, logger=None) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    set_seed(cfg.seed, cfg.train.deterministic)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.train.epochs, cfg.train.warmup_epochs)
    amp_enabled = bool(cfg.train.amp and torch.cuda.is_available())
    scaler = GradScaler(enabled=amp_enabled)
    teacher = ModelEMA(model, decay=cfg.train.ema_decay).to(device) if cfg.train.ema_teacher else None

    best_acc = -1.0
    best_path = Path(run_dir) / "checkpoint_best.pt"
    last_path = Path(run_dir) / "checkpoint_last.pt"
    ensure_dir(run_dir)

    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scaler, device, cfg, teacher)
        scheduler.step()
        val_metrics = evaluate_metrics(model, loaders["val"], device)
        val_acc = val_metrics.get("acc_final", 0.0)
        if logger is not None:
            logger.write_epoch(epoch, train_loss=train_loss, **val_metrics)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)
    return {"best_acc": best_acc}
