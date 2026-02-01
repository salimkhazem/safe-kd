import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.figures import plot_acc_compute, plot_reliability, plot_risk_vs_delta, plot_toy2d_tradeoff, plot_cifar10c_risk
from eval.tables import build_cifar10c_table, build_early_exit_table, build_main_table


def _first_run(results_dir: Path):
    for path in results_dir.rglob("metrics.json"):
        return path.parent
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    if not args.skip_run:
        sweep_cmd = [
            "python",
            "-m",
            "scripts.sweep",
            "--config",
            args.config,
            "--sweep",
            "configs/sweeps/key_subset.yaml",
        ]
        if args.gpus:
            sweep_cmd.extend(["--gpus", args.gpus])
        if not args.full:
            sweep_cmd.append("--fast")
        subprocess.run(sweep_cmd, check=False)

    results_dir = Path("results")
    results_tables = results_dir / "tables"
    results_figures = results_dir / "figures"
    results_tables.mkdir(parents=True, exist_ok=True)
    results_figures.mkdir(parents=True, exist_ok=True)

    build_main_table(results_dir, results_tables / "main_key_subset.csv", results_tables / "main_key_subset.tex")
    build_early_exit_table(results_dir, results_tables / "early_exit_tradeoff.csv", results_tables / "early_exit_tradeoff.tex")
    build_cifar10c_table(results_dir, results_tables / "cifar10c_robustness.csv", results_tables / "cifar10c_robustness.tex")

    run = _first_run(results_dir)
    if run:
        plot_risk_vs_delta(run / "exit_stats.json", results_figures / "risk_vs_delta.pdf")
        plot_acc_compute(run / "exit_stats.json", results_figures / "acc_compute_curve.pdf")
        if (run / "reliability.json").exists():
            plot_reliability(run / "reliability.json", results_figures / "reliability_exit1_exit2_exit3.pdf")
        if (run / "robustness.json").exists():
            plot_cifar10c_risk(run / "robustness.json", results_figures / "cifar10c_risk_vs_delta.pdf")

    # Toy2D quick experiment for theory figure
    if not args.skip_run:
        toy_cmd = [
            "python",
            "-m",
            "scripts.run",
            "--dataset",
            "toy2d",
            "--model",
            "rn50",
            "--method",
            "safe_kd",
            "--seed",
            "0",
            "--fast",
        ]
        if args.gpus:
            toy_cmd.extend(["--gpus", args.gpus])
        subprocess.run(toy_cmd, check=False)
    toy_run = _first_run(results_dir / "toy2d") if (results_dir / "toy2d").exists() else None
    if toy_run and (toy_run / "exit_stats.json").exists():
        plot_toy2d_tradeoff(toy_run / "exit_stats.json", results_figures / "toy2d_tradeoff.pdf")

    # Copy into paper folder
    paper_tables = Path("paper") / "tables"
    paper_figures = Path("paper") / "figures"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)

    for p in results_tables.glob("*.tex"):
        shutil.copy(p, paper_tables / p.name)
    for p in results_figures.glob("*.pdf"):
        shutil.copy(p, paper_figures / p.name)


if __name__ == "__main__":
    main()
