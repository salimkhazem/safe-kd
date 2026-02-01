import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.datasets import build_loaders
from eval.early_exit import naive_risk_curve, safe_risk_curve
from eval.metrics import collect_logits
from eval.robustness import evaluate_cifar10c
from methods.calibration import reliability_bins
from methods.safe_exit import calibrate_thresholds
from models.factory import create_model
from train.engine import evaluate_metrics, train_model
from train.logger import MetricsLogger
from utils.env import save_env
from utils.io import ensure_dir, save_config, save_json


def load_cfg(base_path: str, dataset: str, model: str, method: str):
    cfg = OmegaConf.load(base_path)
    if dataset:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f"configs/datasets/{dataset}.yaml"))
    if model:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f"configs/models/{model}.yaml"))
    if method:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f"configs/methods/{method}.yaml"))
    if "num_classes" in cfg.data:
        cfg.model.num_classes = int(cfg.data.num_classes)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--gpus", default=None, help="GPU ids (e.g. 0 or 0,1). Uses first id for single run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--exit-weights", type=str, default=None)
    parser.add_argument("--conf-head", type=str, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true")
    amp_group.add_argument("--no-amp", action="store_true")
    pretrain_group = parser.add_mutually_exclusive_group()
    pretrain_group.add_argument("--pretrained", action="store_true")
    pretrain_group.add_argument("--no-pretrained", action="store_true")
    det_group = parser.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", action="store_true")
    det_group.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        try:
            gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
            if gpu_ids and torch.cuda.is_available():
                torch.cuda.set_device(gpu_ids[0])
        except Exception:
            pass

    cfg = load_cfg(args.config, args.dataset, args.model, args.method)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.fast:
        cfg.train.epochs = int(cfg.sweep.fast_epochs)
    if cfg.method.name in {"kd", "dkd", "safe_kd"}:
        cfg.train.ema_teacher = True

    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.eval_batch_size is not None:
        cfg.train.eval_batch_size = int(args.eval_batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.weight_decay is not None:
        cfg.train.weight_decay = float(args.weight_decay)
    if args.warmup_epochs is not None:
        cfg.train.warmup_epochs = int(args.warmup_epochs)
    if args.grad_accum_steps is not None:
        cfg.train.grad_accum_steps = int(args.grad_accum_steps)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
    if args.amp:
        cfg.train.amp = True
    if args.no_amp:
        cfg.train.amp = False
    if args.pretrained:
        cfg.model.pretrained = True
    if args.no_pretrained:
        cfg.model.pretrained = False
    if args.deterministic:
        cfg.train.deterministic = True
    if args.no_deterministic:
        cfg.train.deterministic = False
    if args.no_progress:
        cfg.train.show_progress = False
    if args.alpha is not None:
        cfg.method.alpha = float(args.alpha)
    if args.beta is not None:
        cfg.method.beta = float(args.beta)
    if args.temperature is not None:
        cfg.method.temperature = float(args.temperature)
    if args.exit_weights:
        weights = [float(x) for x in args.exit_weights.split(",") if x.strip() != ""]
        if weights:
            cfg.method.exit_weights = weights
    if args.conf_head is not None:
        cfg.method.conf_head = str(args.conf_head)

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    run_dir = Path(cfg.output_dir) / cfg.data.dataset / cfg.model.name / cfg.method.name / f"seed_{cfg.seed}"
    ensure_dir(run_dir)

    save_config(cfg, run_dir / "config.yaml")
    save_env(run_dir / "env.json")

    if cfg.data.dataset == "toy2d":
        cfg.model.name = "toy_mlp"
        cfg.model.pretrained = False
        cfg.data.num_workers = 0
        cfg.train.epochs = int(cfg.train.epochs)

    loaders = build_loaders(cfg)

    model = create_model(cfg.model, cfg.model.num_classes, conf_head_type=cfg.method.conf_head)

    logger = MetricsLogger(str(run_dir / "metrics.json"), str(run_dir / "metrics_per_epoch.jsonl"))
    train_summary = train_model(model, loaders, cfg, str(run_dir), logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_metrics = evaluate_metrics(model, loaders["test"], device)
    logger.update(**test_metrics, **train_summary)
    logger.flush()

    # Calibration and early-exit stats
    cal_logits, cal_labels = collect_logits(model, loaders["cal"], device)
    thresholds = calibrate_thresholds(cal_logits, cal_labels, cfg.calibration.deltas)
    save_json({"thresholds": thresholds}, run_dir / "risk_calibration.json")

    test_logits, test_labels = collect_logits(model, loaders["test"], device)
    safe_stats = safe_risk_curve(test_logits, test_labels, thresholds)
    naive_stats = naive_risk_curve(test_logits, test_labels, cfg.calibration.naive_thresholds)
    save_json({"safe": safe_stats, "naive": naive_stats}, run_dir / "exit_stats.json")

    # Reliability bins
    reliability = {}
    for j, logits in enumerate(test_logits):
        reliability[f"exit{j+1}"] = reliability_bins(logits, test_labels)
    save_json(reliability, run_dir / "reliability.json")

    # Robustness
    if cfg.robustness.enable and cfg.data.dataset == "cifar10":
        robustness = evaluate_cifar10c(cfg, model, device, thresholds)
        save_json(robustness, run_dir / "robustness.json")


if __name__ == "__main__":
    main()
