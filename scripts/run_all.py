import argparse
import os
import subprocess
import sys
import time
from collections import deque
from itertools import product
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_list(values):
    if values is None:
        return None
    if isinstance(values, list) and len(values) == 0:
        return None
    if isinstance(values, list) and len(values) == 1 and "," in values[0]:
        return [v for v in values[0].split(",") if v.strip() != ""]
    return list(values)


def _load_sweep(path: str):
    sweep_cfg = OmegaConf.load(path).sweep
    return (
        list(sweep_cfg.datasets),
        list(sweep_cfg.models),
        list(sweep_cfg.methods),
        [int(s) for s in sweep_cfg.seeds],
    )


def _add_arg(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sweep", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids (e.g. 0,1,2).")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
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
    parser.add_argument("--log-dir", default=None, help="Optional directory to write per-run logs.")
    args = parser.parse_args()

    cfg_base = OmegaConf.load(args.config)
    output_dir = Path(cfg_base.output_dir)

    if args.sweep:
        datasets, models, methods, seeds = _load_sweep(args.sweep)
    else:
        datasets = _parse_list(args.datasets) or []
        models = _parse_list(args.models) or []
        methods = _parse_list(args.methods) or []
        seeds = _parse_list(args.seeds) or []
        if not (datasets and models and methods and seeds):
            raise SystemExit("Provide --sweep or all of --datasets/--models/--methods/--seeds.")
        seeds = [int(s) for s in seeds]

    gpu_ids = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
    if not gpu_ids:
        gpu_ids = ["0"]

    jobs = deque()
    for dataset, model, method, seed in product(datasets, models, methods, seeds):
        run_dir = output_dir / dataset / model / method / f"seed_{seed}"
        if args.skip_existing and (run_dir / "metrics.json").exists():
            continue
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
        jobs.append(cmd)

    if args.dry_run:
        print("Planned runs:", len(jobs))
        for cmd in jobs:
            print(" ".join(cmd))
        return

    available = deque(gpu_ids)
    running = []

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    while jobs or running:
        while jobs and available:
            gpu = available.popleft()
            cmd = jobs.popleft()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_fp = None
            if log_dir:
                name = f"{cmd[cmd.index('--dataset')+1]}_{cmd[cmd.index('--model')+1]}_{cmd[cmd.index('--method')+1]}_seed{cmd[cmd.index('--seed')+1]}.log"
                log_fp = open(log_dir / name, "w", encoding="utf-8")
                proc = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=log_fp)
            else:
                proc = subprocess.Popen(cmd, env=env)
            running.append((proc, gpu, cmd, log_fp))
        still_running = []
        for proc, gpu, cmd, log_fp in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu, cmd, log_fp))
            else:
                if log_fp is not None:
                    log_fp.close()
                available.append(gpu)
        running = still_running
        if jobs and not available:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
