# SAFE-KD: Risk-Controlled Early-Exit Distillation for Vision Backbones

This repo implements SAFE-KD with a universal multi-exit wrapper, DKD-based hierarchical distillation, and conformal risk-controlled early-exit calibration. It supports multiple datasets, backbones, and reproducible runs with one-command reproduction.
```LaTex
Early-exit neural networks can significantly reduce inference cost by allowing “easy” inputs to terminate at intermediate layers, but practical deployment hinges on deciding when it is safe to exit. We introduce SAFE-KD, a universal multi-exit wrapper for modern vision backbones (CNNs and Transformers) that couples hierarchical knowledge distillation with post-hoc risk-controlled exiting. Architecturally, SAFE-KD attaches lightweight classifier branches at multiple depths, producing calibrated predictions at each exit with negligible overhead. During training, we distill a strong teacher into all exits using Decoupled Knowledge Distillation (DKD) while enforcing deep-to-shallow consistency, improving intermediate accuracy and stability. At inference, we replace heuristic confidence thresholds with a conformal risk-control calibration step that sets exit thresholds to satisfy a user-specified risk level under standard exchangeability assumptions. Across a suite of public vision datasets and backbones, SAFE-KD yields favorable accuracy–compute trade-offs, improved calibration, and robust performance under common corruptions, enabling fast inference with principled quality control.
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Datasets
By default, torchvision datasets are downloaded to `data/`. CIFAR-10-C is optional; if not found, synthetic corruptions are used.
If you are offline, pre-download datasets into `data/` to avoid download failures.

Supported datasets:
- CIFAR-10, CIFAR-100, STL-10
- Oxford-IIIT Pet, Flowers102, FGVC Aircraft
- CIFAR-10-C (optional)
- Toy 2D synthetic

## Quickstart (single run)
```bash
python -m scripts.run --dataset cifar10 --model rn50 --method safe_kd --seed 0
```

## GPU selection
Single run on a specific GPU:
```bash
python -m scripts.run --dataset cifar10 --model rn50 --method safe_kd --seed 0 --gpus 0
```
Multi-GPU training is not implemented; use `--gpus` to select a single GPU per process.

## Override batch size / epochs / LR
```bash
python -m scripts.run --dataset cifar10 --model rn50 --method safe_kd --seed 0 \
  --batch-size 128 --eval-batch-size 256 --epochs 50 --lr 1e-4
```

## Sweep (key subset)
```bash
python -m scripts.sweep --sweep configs/sweeps/key_subset.yaml --fast
```

## One-command reproduction
```bash
python -m scripts.reproduce
```
Add `--full` to run full epochs:
```bash
python -m scripts.reproduce --full
```

## Run all experiments (GPU scheduler)
```bash
python -m scripts.run_all --sweep configs/sweeps/full_benchmark.yaml --gpus 0,1,2 --skip-existing
```

## Collect results
```bash
python -m scripts.collect_results --results-dir results --output results/summary.csv --format both
```

## Toy 2D experiment
```bash
python -m scripts.run --dataset toy2d --model rn50 --method safe_kd --seed 0 --fast
```

## Expected artifacts
Each run writes to:
```
results/<dataset>/<model>/<method>/seed_<s>/
  config.yaml
  env.json
  metrics.json
  metrics_per_epoch.jsonl
  checkpoint_best.pt
  checkpoint_last.pt
  exit_stats.json
  risk_calibration.json
  reliability.json
  robustness.json (if enabled)
```

Aggregate outputs:
```
results/tables/main_key_subset.csv
results/tables/early_exit_tradeoff.csv
results/tables/cifar10c_robustness.csv
results/figures/risk_vs_delta.pdf
results/figures/acc_compute_curve.pdf
results/figures/reliability_exit1_exit2_exit3.pdf
```

The reproduction script copies tables/figures into `paper/` and the LaTeX skeleton compiles as-is.

## Notes on reproducibility
- Deterministic splits are saved under `data/splits/`.
- Seeds are applied to Python/NumPy/PyTorch and DataLoader workers.
- Configs and environment info are logged per run.
- If you see `torch_shm_manager` errors, keep `data.num_workers: 0` (default in `configs/base.yaml`).

## Smoke test
```bash
python -m scripts.run --dataset cifar10 --model rn50 --method safe_kd --seed 0 --fast
pytest
```
