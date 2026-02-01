import argparse
import statistics
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.figures import plot_toy2d_sota_tradeoff


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def _run_config(config_path: Path, seeds, fast: bool):
    for seed in seeds:
        cmd = [
            "python",
            "-m",
            "scripts.run",
            "--config",
            str(config_path),
            "--dataset",
            "toy2d",
            "--model",
            "rn50",
            "--seed",
            str(seed),
        ]
        if fast:
            cmd.append("--fast")
        subprocess.run(cmd, check=False)


def _summarize_config(output_dir: Path, seeds, delta: float):
    acc_final = []
    acc_delta = []
    risk_delta = []
    compute = []
    for seed in seeds:
        run_dir = output_dir / "toy2d" / "rn50" / "safe_kd" / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        exit_path = run_dir / "exit_stats.json"
        if not metrics_path.exists() or not exit_path.exists():
            continue
        metrics = OmegaConf.load(metrics_path)
        exit_stats = OmegaConf.load(exit_path)
        key = str(delta)
        if key not in exit_stats.get("safe", {}):
            continue
        stats = exit_stats["safe"][key]
        acc_final.append(float(metrics.get("acc_exit3", metrics.get("acc_final", 0.0))))
        acc_delta.append(float(stats.get("overall_acc", 0.0)))
        risk_delta.append(float(stats.get("overall_risk", 0.0)))
        compute.append(float(stats.get("expected_compute", 0.0)))

    acc_final_mean, acc_final_std = _mean_std(acc_final)
    acc_delta_mean, acc_delta_std = _mean_std(acc_delta)
    risk_delta_mean, _ = _mean_std(risk_delta)
    compute_mean, _ = _mean_std(compute)
    return {
        "acc_final_mean": acc_final_mean,
        "acc_final_std": acc_final_std,
        "acc_delta_mean": acc_delta_mean,
        "acc_delta_std": acc_delta_std,
        "risk_delta_mean": risk_delta_mean,
        "expected_compute_mean": compute_mean,
    }


def _safe_tag(value: float) -> str:
    text = f"{value:.3f}".replace(".", "p")
    return text.rstrip("0").rstrip("p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    seeds = [0, 1, 2]
    alpha_grid = [1.0]
    beta_grid = [0.25, 0.5, 1.0, 2.0]
    temp_grid = [2.0, 4.0, 8.0]

    base_cfg = OmegaConf.load(args.config)

    results_root = Path("results_tune")
    configs_dir = results_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for alpha in alpha_grid:
        for beta in beta_grid:
            for temp in temp_grid:
                tag = f"safe_kd_a{_safe_tag(alpha)}_b{_safe_tag(beta)}_t{_safe_tag(temp)}"
                output_dir = results_root / tag
                cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
                cfg.method.name = "safe_kd"
                cfg.method.alpha = float(alpha)
                cfg.method.beta = float(beta)
                cfg.method.temperature = float(temp)
                cfg.output_dir = str(output_dir)
                cfg.exp_name = tag

                cfg_path = configs_dir / f"{tag}.yaml"
                OmegaConf.save(cfg, cfg_path)

                if not args.skip_run:
                    _run_config(cfg_path, seeds, args.fast)

                summary = _summarize_config(output_dir, seeds, args.delta)
                summary.update({"tag": tag, "alpha": alpha, "beta": beta, "temperature": temp})
                summaries.append(summary)

    if not summaries:
        return

    # Select best config: maximize acc_delta_mean with risk <= delta + 0.01.
    tol = 0.01
    feasible = [s for s in summaries if s["risk_delta_mean"] <= args.delta + tol]
    if feasible:
        best = max(feasible, key=lambda s: (s["acc_delta_mean"], -s["expected_compute_mean"]))
    else:
        best = min(summaries, key=lambda s: (abs(s["risk_delta_mean"] - args.delta), -s["acc_delta_mean"]))

    # Write summary CSV.
    summary_path = results_root / "toy2d_safe_kd_tune_summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            "tag,alpha,beta,temperature,acc_final_mean,acc_final_std,acc_delta_mean,acc_delta_std,risk_delta_mean,expected_compute_mean\n"
        )
        for s in summaries:
            f.write(
                f"{s['tag']},{s['alpha']},{s['beta']},{s['temperature']},"
                f"{s['acc_final_mean']:.6f},{s['acc_final_std']:.6f},"
                f"{s['acc_delta_mean']:.6f},{s['acc_delta_std']:.6f},"
                f"{s['risk_delta_mean']:.6f},{s['expected_compute_mean']:.6f}\n"
            )

    # Copy best run into results/ for plotting alongside baselines.
    tuned_method = f"safe_kd_tuned_{best['tag'].replace('safe_kd_', '')}"
    dest_root = Path("results") / "toy2d" / "rn50" / tuned_method
    if dest_root.exists():
        # Avoid overwriting user data.
        suffix = 2
        while (Path("results") / "toy2d" / "rn50" / f"{tuned_method}_v{suffix}").exists():
            suffix += 1
        tuned_method = f"{tuned_method}_v{suffix}"
        dest_root = Path("results") / "toy2d" / "rn50" / tuned_method

    for seed in seeds:
        src_dir = results_root / best["tag"] / "toy2d" / "rn50" / "safe_kd" / f"seed_{seed}"
        dst_dir = dest_root / f"seed_{seed}"
        if src_dir.exists():
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-r", str(src_dir), str(dst_dir)], check=False)

    # Plot combined figure.
    fig_path = Path("results") / "figures" / "toy2d_sota_tradeoff_tuned.pdf"
    baseline_methods = ["safe_kd", "dkd", "kd", "multiexit", "erm"]
    plot_toy2d_sota_tradeoff("results", fig_path, method_allowlist=baseline_methods + [tuned_method])

    # Copy tuned figure into paper/figures.
    paper_figures = Path("paper") / "figures"
    paper_figures.mkdir(parents=True, exist_ok=True)
    if fig_path.exists():
        subprocess.run(["cp", str(fig_path), str(paper_figures / fig_path.name)], check=False)

    # Print best config summary.
    print("Best SAFE-KD config:")
    print(best)


if __name__ == "__main__":
    main()
