import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _collect_runs(results_dir: Path, dataset: str, model: str, methods: List[str]):
    runs = {}
    for method in methods:
        method_dir = results_dir / dataset / model / method
        if not method_dir.exists():
            continue
        for seed_dir in method_dir.glob("seed_*"):
            if (seed_dir / "metrics.json").exists():
                runs.setdefault(method, []).append(seed_dir)
    return runs


def _mean_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values))


def build_highlight_table(
    results_dir: Path,
    dataset: str,
    model: str,
    methods: List[str],
    delta: float,
    out_path: Path,
):
    runs = _collect_runs(results_dir, dataset, model, methods)
    rows = []
    for method in methods:
        acc_final = []
        acc_delta = []
        risk_delta = []
        compute_delta = []
        for seed_dir in runs.get(method, []):
            with open(seed_dir / "metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
            acc_final.append(metrics.get("acc_exit3", metrics.get("acc_final", 0.0)))
            exit_path = seed_dir / "exit_stats.json"
            if exit_path.exists():
                data = json.load(open(exit_path))
                key = str(delta)
                stats = data.get("safe", {}).get(key)
                if stats:
                    acc_delta.append(stats.get("overall_acc", 0.0))
                    risk_delta.append(stats.get("overall_risk", 0.0))
                    compute_delta.append(stats.get("expected_compute", 0.0))
        af_m, af_s = _mean_std(acc_final)
        ad_m, ad_s = _mean_std(acc_delta)
        rd_m, rd_s = _mean_std(risk_delta)
        cd_m, cd_s = _mean_std(compute_delta)
        rows.append(
            dict(
                method=method,
                acc_final=(af_m, af_s),
                acc_delta=(ad_m, ad_s),
                risk_delta=(rd_m, rd_s),
                compute_delta=(cd_m, cd_s),
            )
        )

    safe_row = next((r for r in rows if r["method"] == "safe_kd"), None)
    baseline_acc = [r["acc_delta"][0] for r in rows if r["method"] != "safe_kd"]
    baseline_risk = [r["risk_delta"][0] for r in rows if r["method"] != "safe_kd"]
    best_acc = max(baseline_acc) if baseline_acc else None
    best_risk = min(baseline_risk) if baseline_risk else None

    def fmt(val):
        return f"{val:.3f}"

    def fmt_pm(mean, std):
        if std == 0.0:
            return fmt(mean)
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    lines = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & Final Acc & Acc@$\\delta$ & Risk@$\\delta$ & Compute@$\\delta$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        name = r["method"]
        af = fmt_pm(*r["acc_final"])
        ad = fmt_pm(*r["acc_delta"])
        rd = fmt_pm(*r["risk_delta"])
        cd = fmt_pm(*r["compute_delta"])
        if name == "safe_kd" and safe_row is not None and best_acc is not None and best_risk is not None:
            ad = f"{ad} ({r['acc_delta'][0]-best_acc:+.3f})"
            rd = f"{rd} ({r['risk_delta'][0]-best_risk:+.3f})"
        if name == "safe_kd":
            name = "\\textbf{SAFE-KD}"
            af = f"\\textbf{{{af}}}"
            ad = f"\\textbf{{{ad}}}"
            rd = f"\\textbf{{{rd}}}"
            cd = f"\\textbf{{{cd}}}"
        lines.append(f"{name} & {af} & {ad} & {rd} & {cd} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def plot_method_curves(
    results_dir: Path,
    dataset: str,
    model: str,
    methods: List[str],
    out_acc_compute: Path,
    out_risk_delta: Path,
):
    runs = _collect_runs(results_dir, dataset, model, methods)
    method_curves: Dict[str, Dict[str, List[float]]] = {}
    for method, seed_dirs in runs.items():
        per_delta = {}
        for seed_dir in seed_dirs:
            exit_path = seed_dir / "exit_stats.json"
            if not exit_path.exists():
                continue
            data = json.load(open(exit_path))
            for delta, stats in data.get("safe", {}).items():
                entry = per_delta.setdefault(delta, {"acc": [], "risk": [], "compute": []})
                entry["acc"].append(stats.get("overall_acc", 0.0))
                entry["risk"].append(stats.get("overall_risk", 0.0))
                entry["compute"].append(stats.get("expected_compute", 0.0))
        method_curves[method] = per_delta

    colors = {
        "safe_kd": "#D62728",
        "dkd": "#1F77B4",
        "kd": "#2CA02C",
        "multiexit": "#FF7F0E",
        "erm": "#7F7F7F",
    }

    # Accuracy vs compute
    plt.figure(figsize=(4.2, 3.1))
    for method in methods:
        per_delta = method_curves.get(method, {})
        if not per_delta:
            continue
        deltas = sorted([float(k) for k in per_delta.keys()])
        acc = []
        comp = []
        for d in deltas:
            stats = per_delta[str(d)]
            acc.append(float(np.mean(stats["acc"])))
            comp.append(float(np.mean(stats["compute"])))
        lw = 2.6 if method == "safe_kd" else 1.6
        plt.plot(comp, acc, marker="o", label=method, linewidth=lw, color=colors.get(method, None))
    plt.xlabel("expected compute")
    plt.ylabel("accuracy")
    plt.title(f"{dataset}-{model}: Acc vs compute")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_acc_compute.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_acc_compute, dpi=300)
    plt.close()

    # Risk vs delta
    plt.figure(figsize=(4.2, 3.1))
    for method in methods:
        per_delta = method_curves.get(method, {})
        if not per_delta:
            continue
        deltas = sorted([float(k) for k in per_delta.keys()])
        risks = []
        for d in deltas:
            stats = per_delta[str(d)]
            risks.append(float(np.mean(stats["risk"])))
        lw = 2.6 if method == "safe_kd" else 1.6
        plt.plot(deltas, risks, marker="o", label=method, linewidth=lw, color=colors.get(method, None))
    if deltas:
        plt.plot(deltas, deltas, linestyle="--", color="gray", linewidth=1.0, label="risk = delta")
    plt.xlabel("delta")
    plt.ylabel("risk")
    plt.title(f"{dataset}-{model}: Risk vs delta")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_risk_delta.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_risk_delta, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="rn50")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--methods", nargs="*", default=["erm", "multiexit", "kd", "dkd", "safe_kd"])
    parser.add_argument("--out-table", default="paper/tables/highlight_cifar10_rn50.tex")
    parser.add_argument("--out-acc-compute", default="paper/figures/highlight_cifar10_rn50_acc_compute.pdf")
    parser.add_argument("--out-risk-delta", default="paper/figures/highlight_cifar10_rn50_risk_delta.pdf")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    build_highlight_table(results_dir, args.dataset, args.model, args.methods, args.delta, Path(args.out_table))
    plot_method_curves(
        results_dir,
        args.dataset,
        args.model,
        args.methods,
        Path(args.out_acc_compute),
        Path(args.out_risk_delta),
    )


if __name__ == "__main__":
    main()
