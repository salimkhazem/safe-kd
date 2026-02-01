import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.figures import plot_toy2d_sota_tradeoff
from eval.tables import build_toy2d_sota_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sweep", default="configs/sweeps/toy2d_sota.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if not args.skip_run:
        sweep_cmd = [
            "python",
            "-m",
            "scripts.sweep",
            "--config",
            args.config,
            "--sweep",
            args.sweep,
        ]
        if args.fast:
            sweep_cmd.append("--fast")
        subprocess.run(sweep_cmd, check=False)

    results_dir = Path("results")
    results_tables = results_dir / "tables"
    results_figures = results_dir / "figures"
    results_tables.mkdir(parents=True, exist_ok=True)
    results_figures.mkdir(parents=True, exist_ok=True)

    build_toy2d_sota_table(
        results_dir,
        results_tables / "toy2d_sota.csv",
        results_tables / "toy2d_sota.tex",
        delta=0.05,
    )
    plot_toy2d_sota_tradeoff(results_dir, results_figures / "toy2d_sota_tradeoff.pdf")

    # Copy into paper folder for convenience.
    paper_tables = Path("paper") / "tables"
    paper_figures = Path("paper") / "figures"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)
    for p in results_tables.glob("toy2d_sota.*"):
        shutil.copy(p, paper_tables / p.name)
    for p in results_figures.glob("toy2d_sota_tradeoff.pdf"):
        shutil.copy(p, paper_figures / p.name)


if __name__ == "__main__":
    main()
