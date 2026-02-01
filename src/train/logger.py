from typing import Dict

from utils.io import append_jsonl, save_json


class MetricsLogger:
    def __init__(self, metrics_path: str, per_epoch_path: str):
        self.metrics_path = metrics_path
        self.per_epoch_path = per_epoch_path
        self.metrics: Dict[str, float] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.metrics[k] = float(v)

    def write_epoch(self, epoch: int, **kwargs):
        payload = {"epoch": epoch}
        payload.update({k: float(v) for k, v in kwargs.items()})
        append_jsonl(payload, self.per_epoch_path)

    def flush(self):
        save_json(self.metrics, self.metrics_path)
