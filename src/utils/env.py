import json
import platform
from typing import Any, Dict

import torch

from utils.io import find_git_hash


def collect_env() -> Dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
    env["git_hash"] = find_git_hash()
    return env


def save_env(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect_env(), f, indent=2)
