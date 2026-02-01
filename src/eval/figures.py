from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import matplotlib.pyplot as plt
import json


def plot_reliability(reliability_path: str, out_path: str):
    with open(reliability_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    plt.figure(figsize=(4, 3))
    for exit_name, bins in data.items():
        bin_conf = bins.get("bin_conf", [])
        bin_acc = bins.get("bin_acc", [])
        if len(bin_conf) == 0:
            continue
        plt.plot(bin_conf, bin_acc, marker="o", label=exit_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.title("Reliability diagram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_risk_vs_delta(exit_stats_path: str, out_path: str):
    with open(exit_stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    safe = data.get("safe", {})
    if not safe:
        return None
    deltas = sorted([float(k) for k in safe.keys()])
    risks = [safe[str(d)]["overall_risk"] for d in deltas]

    plt.figure(figsize=(4, 3))
    plt.plot(deltas, risks, marker="o", label="SAFE")
    plt.xlabel("delta")
    plt.ylabel("overall risk")
    plt.title("Risk vs delta")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_acc_compute(exit_stats_path: str, out_path: str):
    with open(exit_stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    safe = data.get("safe", {})
    if not safe:
        return None
    deltas = sorted([float(k) for k in safe.keys()])
    acc = [safe[str(d)]["overall_acc"] for d in deltas]
    comp = [safe[str(d)]["expected_compute"] for d in deltas]

    plt.figure(figsize=(4, 3))
    plt.plot(comp, acc, marker="o", label="SAFE")
    plt.xlabel("expected compute (depth fraction)")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs compute")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_toy2d_tradeoff(exit_stats_path: str, out_path: str):
    with open(exit_stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    safe = data.get("safe", {})
    if not safe:
        return None
    deltas = sorted([float(k) for k in safe.keys()])
    risks = [safe[str(d)]["overall_risk"] for d in deltas]
    compute = [safe[str(d)]["expected_compute"] for d in deltas]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    axes[0].plot(deltas, risks, marker="o")
    axes[0].set_xlabel("delta")
    axes[0].set_ylabel("risk")
    axes[0].set_title("Risk vs delta")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(deltas, compute, marker="o", color="tab:orange")
    axes[1].set_xlabel("delta")
    axes[1].set_ylabel("expected compute")
    axes[1].set_title("Compute vs delta")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cifar10c_risk(robustness_path: str, out_path: str):
    with open(robustness_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    deltas = []
    risks = []
    for k, v in data.items():
        if k.startswith("risk_delta_"):
            deltas.append(float(k.replace("risk_delta_", "")))
            risks.append(float(v))
    if not deltas:
        return None
    order = sorted(range(len(deltas)), key=lambda i: deltas[i])
    deltas = [deltas[i] for i in order]
    risks = [risks[i] for i in order]
    plt.figure(figsize=(4, 3))
    plt.plot(deltas, risks, marker="o")
    plt.xlabel("delta")
    plt.ylabel("risk under corruption")
    plt.title("CIFAR-10-C risk vs delta")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def _collect_toy2d_runs(results_dir: Path) -> List[Path]:
    dataset_dir = results_dir / "toy2d"
    if not dataset_dir.exists():
        return []
    runs = []
    for model_dir in dataset_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for method_dir in model_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.iterdir():
                if seed_dir.is_dir() and (seed_dir / "exit_stats.json").exists():
                    runs.append(seed_dir)
    return runs


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def plot_toy2d_sota_tradeoff(results_dir: str, out_path: str, method_allowlist: List[str] | None = None):
    results_dir = Path(results_dir)
    runs = _collect_toy2d_runs(results_dir)
    if not runs:
        return None

    # Aggregate per-method curves across seeds.
    method_stats = {}
    for run in runs:
        method = run.parent.name
        if method_allowlist is not None and method not in method_allowlist:
            continue
        with open(run / "exit_stats.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        safe = data.get("safe", {})
        if not safe:
            continue
        for delta_str, stats in safe.items():
            delta = float(delta_str)
            entry = method_stats.setdefault(method, {}).setdefault(
                delta, {"acc": [], "compute": [], "risk": []}
            )
            entry["acc"].append(float(stats.get("overall_acc", 0.0)))
            entry["compute"].append(float(stats.get("expected_compute", 0.0)))
            entry["risk"].append(float(stats.get("overall_risk", 0.0)))

    if not method_stats:
        return None

    label_map = {
        "safe_kd": "SAFE-KD (ours)",
        "dkd": "DKD",
        "kd": "KD",
        "multiexit": "Multi-Exit",
        "erm": "ERM",
    }
    palette = {
        "safe_kd": "#D62728",
        "safe_kd_tuned": "#8B0000",
        "dkd": "#1F77B4",
        "kd": "#2CA02C",
        "multiexit": "#FF7F0E",
        "erm": "#7F7F7F",
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    def _method_key(name: str) -> int:
        if name.startswith("safe_kd_tuned"):
            return 0
        if name == "safe_kd":
            return 1
        if name == "dkd":
            return 2
        if name == "kd":
            return 3
        if name == "multiexit":
            return 4
        if name == "erm":
            return 5
        return 6

    methods_sorted = sorted(method_stats.keys(), key=_method_key)

    # Accuracy vs compute.
    for method in methods_sorted:
        if method not in method_stats:
            continue
        deltas = sorted(method_stats[method].keys())
        acc = []
        comp = []
        for d in deltas:
            acc_mean, _ = _mean_std(method_stats[method][d]["acc"])
            comp_mean, _ = _mean_std(method_stats[method][d]["compute"])
            acc.append(acc_mean)
            comp.append(comp_mean)
        tuned = method.startswith("safe_kd_tuned")
        color = palette.get("safe_kd_tuned" if tuned else method, "#444444")
        label = label_map.get(method, method)
        if tuned:
            suffix = method.replace("safe_kd_tuned_", "")
            label = f"SAFE-KD (tuned {suffix})" if suffix != "safe_kd_tuned" else "SAFE-KD (tuned)"
        lw = 2.8 if tuned or method == "safe_kd" else 1.6
        alpha = 1.0 if tuned or method == "safe_kd" else 0.85
        axes[0].plot(comp, acc, marker="o", color=color, label=label, linewidth=lw, alpha=alpha)

    axes[0].set_xlabel("expected compute (depth fraction)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Toy2D: Accuracy vs compute")
    axes[0].grid(True, alpha=0.3)

    # Risk vs delta.
    for method in methods_sorted:
        if method not in method_stats:
            continue
        deltas = sorted(method_stats[method].keys())
        risks = []
        for d in deltas:
            risk_mean, _ = _mean_std(method_stats[method][d]["risk"])
            risks.append(risk_mean)
        tuned = method.startswith("safe_kd_tuned")
        color = palette.get("safe_kd_tuned" if tuned else method, "#444444")
        label = label_map.get(method, method)
        if tuned:
            suffix = method.replace("safe_kd_tuned_", "")
            label = f"SAFE-KD (tuned {suffix})" if suffix != "safe_kd_tuned" else "SAFE-KD (tuned)"
        lw = 2.8 if tuned or method == "safe_kd" else 1.6
        alpha = 1.0 if tuned or method == "safe_kd" else 0.85
        axes[1].plot(deltas, risks, marker="o", color=color, label=label, linewidth=lw, alpha=alpha)

    if method_stats:
        all_deltas = sorted({d for m in method_stats.values() for d in m.keys()})
        if all_deltas:
            axes[1].plot(all_deltas, all_deltas, linestyle="--", color="gray", linewidth=1.0, label="risk = delta")

    axes[1].set_xlabel("delta")
    axes[1].set_ylabel("overall risk")
    axes[1].set_title("Toy2D: Risk control")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
