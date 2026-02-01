import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sweep", default="configs/sweeps/key_subset.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--gpus", default=None)
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

    sweep_cfg = OmegaConf.load(args.sweep).sweep
    datasets = sweep_cfg.datasets
    models = sweep_cfg.models
    methods = sweep_cfg.methods
    seeds = sweep_cfg.seeds

    def _add_arg(cmd, flag, value):
        if value is not None:
            cmd.extend([flag, str(value)])

    for dataset, model, method, seed in product(datasets, models, methods, seeds):
        cmd = [
            "python",
            "-m",
            "scripts.run",
            "--config",
            args.config,
            "--dataset",
            dataset,
            "--model",
            model,
            "--method",
            method,
            "--seed",
            str(seed),
        ]
        _add_arg(cmd, "--gpus", args.gpus)
        _add_arg(cmd, "--epochs", args.epochs)
        _add_arg(cmd, "--batch-size", args.batch_size)
        _add_arg(cmd, "--eval-batch-size", args.eval_batch_size)
        _add_arg(cmd, "--lr", args.lr)
        _add_arg(cmd, "--weight-decay", args.weight_decay)
        _add_arg(cmd, "--warmup-epochs", args.warmup_epochs)
        _add_arg(cmd, "--grad-accum-steps", args.grad_accum_steps)
        _add_arg(cmd, "--num-workers", args.num_workers)
        _add_arg(cmd, "--alpha", args.alpha)
        _add_arg(cmd, "--beta", args.beta)
        _add_arg(cmd, "--temperature", args.temperature)
        _add_arg(cmd, "--exit-weights", args.exit_weights)
        _add_arg(cmd, "--conf-head", args.conf_head)
        if args.amp:
            cmd.append("--amp")
        if args.no_amp:
            cmd.append("--no-amp")
        if args.pretrained:
            cmd.append("--pretrained")
        if args.no_pretrained:
            cmd.append("--no-pretrained")
        if args.deterministic:
            cmd.append("--deterministic")
        if args.no_deterministic:
            cmd.append("--no-deterministic")
        if args.no_progress:
            cmd.append("--no-progress")
        if args.fast:
            cmd.append("--fast")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
