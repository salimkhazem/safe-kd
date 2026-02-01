from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import pandas as pd


def _find_runs(results_dir: Path):
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


def build_main_table(results_dir: str, out_csv: str, out_tex: str):
    results_dir = Path(results_dir)
    rows = []
    for run in _find_runs(results_dir):
        parts = run.parts[-4:]
        dataset, model, method, seed = parts
        with open(run / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "seed": seed,
                "acc_final": metrics.get("acc_exit3", metrics.get("acc_final", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    agg = df.groupby(["dataset", "model", "method"]).agg({"acc_final": ["mean", "std"]})
    agg.columns = ["acc_mean", "acc_std"]
    agg = agg.reset_index()
    agg.to_csv(out_csv, index=False)

    # Build a compact, paper-ready LaTeX table (wide format).
    method_order = ["erm", "multiexit", "kd", "dkd", "safe_kd"]
    method_label = {
        "erm": "ERM",
        "multiexit": "MultiExit",
        "kd": "KD",
        "dkd": "DKD",
        "safe_kd": "SAFE-KD",
    }
    dataset_order = ["cifar10", "cifar100", "stl10", "pets", "flowers102", "aircraft"]
    model_order = ["rn50", "mnv3s", "effb0", "convnext_t", "vit_s", "swin_t"]
    dataset_label = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "stl10": "STL-10",
        "pets": "Pets",
        "flowers102": "Flowers102",
        "aircraft": "Aircraft",
    }
    model_label = {
        "rn50": "ResNet-50",
        "mnv3s": "MobileNetV3-S",
        "effb0": "EffNet-B0",
        "convnext_t": "ConvNeXt-T",
        "vit_s": "ViT-S",
        "swin_t": "Swin-T",
    }

    agg = agg[agg["dataset"] != "toy2d"]

    def _fmt(mean: float | None, std: float | None) -> str:
        if mean is None or pd.isna(mean):
            return "--"
        val = mean * 100.0
        if std is None or pd.isna(std) or std == 0.0:
            return f"{val:.2f}"
        return f"{val:.2f} $\\pm$ {std * 100.0:.2f}"

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Final-exit accuracy (\\%) across datasets and backbones.}")
    lines.append("\\label{tab:main-results}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    header = ["Dataset", "Backbone"] + [method_label[m] for m in method_order]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    last_dataset = None
    for dataset in dataset_order + [d for d in agg["dataset"].unique() if d not in dataset_order]:
        subset_d = agg[agg["dataset"] == dataset]
        if subset_d.empty:
            continue
        for model in model_order + [m for m in subset_d["model"].unique() if m not in model_order]:
            subset = subset_d[subset_d["model"] == model]
            if subset.empty:
                continue
            if last_dataset is not None and dataset != last_dataset:
                lines.append("\\midrule")
            last_dataset = dataset

            row_vals = {}
            for method in method_order:
                rec = subset[subset["method"] == method]
                if rec.empty:
                    row_vals[method] = (None, None)
                else:
                    row_vals[method] = (rec["acc_mean"].iloc[0], rec["acc_std"].iloc[0])

            best = None
            for method in method_order:
                mean = row_vals[method][0]
                if mean is None or pd.isna(mean):
                    continue
                best = mean if best is None else max(best, mean)

            cells = []
            for method in method_order:
                mean, std = row_vals[method]
                cell = _fmt(mean, std)
                if best is not None and mean is not None and not pd.isna(mean) and abs(mean - best) < 1e-12:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)

            ds = dataset_label.get(dataset, dataset)
            md = model_label.get(model, model)
            lines.append(f"{ds} & {md} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    Path(out_tex).write_text("\n".join(lines), encoding="utf-8")
    return agg


def build_early_exit_table(results_dir: str, out_csv: str, out_tex: str, delta: float = 0.05):
    results_dir = Path(results_dir)
    rows = []
    for run in _find_runs(results_dir):
        exit_path = run / "exit_stats.json"
        if not exit_path.exists():
            continue
        parts = run.parts[-4:]
        dataset, model, method, seed = parts
        with open(exit_path, "r", encoding="utf-8") as f:
            exit_stats = json.load(f)
        key = str(delta)
        if key in exit_stats.get("safe", {}):
            stats = exit_stats["safe"][key]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "seed": seed,
                    "expected_compute": stats.get("expected_compute", 0.0),
                    "exit1_rate": stats.get("exit_rates", [0, 0, 0])[0],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    numeric_cols = df.select_dtypes(include="number").columns
    agg = df.groupby(["dataset", "model", "method"])[numeric_cols].mean().reset_index()
    agg.to_csv(out_csv, index=False)
    agg.to_latex(out_tex, index=False, float_format="%.4f")
    return agg


def build_cifar10c_table(results_dir: str, out_csv: str, out_tex: str):
    results_dir = Path(results_dir)
    rows = []
    for run in _find_runs(results_dir):
        rob_path = run / "robustness.json"
        if not rob_path.exists():
            continue
        parts = run.parts[-4:]
        dataset, model, method, seed = parts
        if dataset != "cifar10":
            continue
        with open(rob_path, "r", encoding="utf-8") as f:
            rob = json.load(f)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "seed": seed,
                "mean_corruption_acc": rob.get("mean_corruption_acc", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    numeric_cols = df.select_dtypes(include="number").columns
    agg = df.groupby(["dataset", "model", "method"])[numeric_cols].mean().reset_index()
    agg.to_csv(out_csv, index=False)
    agg.to_latex(out_tex, index=False, float_format="%.4f")
    return agg


def build_toy2d_sota_table(results_dir: str, out_csv: str, out_tex: str, delta: float = 0.05):
    results_dir = Path(results_dir)
    rows = []
    for run in _find_runs(results_dir):
        parts = run.parts[-4:]
        dataset, model, method, seed = parts
        if dataset != "toy2d":
            continue
        exit_path = run / "exit_stats.json"
        if not exit_path.exists():
            continue
        with open(exit_path, "r", encoding="utf-8") as f:
            exit_stats = json.load(f)
        metrics = {}
        if (run / "metrics.json").exists():
            with open(run / "metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
        key = str(delta)
        if key not in exit_stats.get("safe", {}):
            continue
        stats = exit_stats["safe"][key]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "seed": seed,
                "acc_final": metrics.get("acc_exit3", metrics.get("acc_final", 0.0)),
                "acc_delta": stats.get("overall_acc", 0.0),
                "risk_delta": stats.get("overall_risk", 0.0),
                "expected_compute": stats.get("expected_compute", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    agg = df.groupby(["dataset", "model", "method"]).agg(
        {
            "acc_final": ["mean", "std"],
            "acc_delta": ["mean", "std"],
            "risk_delta": "mean",
            "expected_compute": "mean",
        }
    )
    agg.columns = [
        "acc_final_mean",
        "acc_final_std",
        "acc_delta_mean",
        "acc_delta_std",
        "risk_delta_mean",
        "expected_compute_mean",
    ]
    agg = agg.reset_index()
    agg.to_csv(out_csv, index=False)
    agg.to_latex(out_tex, index=False, float_format="%.4f")
    return agg
