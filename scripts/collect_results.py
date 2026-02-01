import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _find_runs(results_dir: Path) -> List[Path]:
    runs = []
    for dataset in results_dir.iterdir():
        if not dataset.is_dir():
            continue
        for model in dataset.iterdir():
            if not model.is_dir():
                continue
            for method in model.iterdir():
                if not method.is_dir():
                    continue
                for seed in method.iterdir():
                    if seed.is_dir() and (seed / "metrics.json").exists():
                        runs.append(seed)
    return runs


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    parser.add_argument("--format", choices=["csv", "json", "both"], default="both")
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for run in _find_runs(results_dir):
        parts = run.parts[-4:]
        dataset, model, method, seed = parts
        metrics = _read_json(run / "metrics.json")
        acc_final = metrics.get("acc_final", metrics.get("acc_exit3", 0.0))
        nll_final = metrics.get("nll_exit3", metrics.get("nll_exit1", 0.0))
        ece_final = metrics.get("ece_exit3", metrics.get("ece_exit1", 0.0))
        exit_stats = {}
        exit_path = run / "exit_stats.json"
        if exit_path.exists():
            exit_stats = _read_json(exit_path)
        delta_key = str(args.delta)
        safe_stats = exit_stats.get("safe", {}).get(delta_key, {})
        row = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "seed": seed,
            "acc_final": acc_final,
            "nll_final": nll_final,
            "ece_final": ece_final,
            "best_acc": metrics.get("best_acc", 0.0),
            "acc_delta": safe_stats.get("overall_acc", None),
            "risk_delta": safe_stats.get("overall_risk", None),
            "expected_compute": safe_stats.get("expected_compute", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format in {"csv", "both"}:
        df.to_csv(output_path, index=False)
    if args.format in {"json", "both"}:
        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    # Aggregated summary (mean/std) for numeric columns.
    numeric_cols = df.select_dtypes(include="number").columns
    agg = df.groupby(["dataset", "model", "method"])[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    agg_path = output_path.with_name("summary_aggregated.csv")
    agg.to_csv(agg_path, index=False)


if __name__ == "__main__":
    main()
