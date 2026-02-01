from __future__ import annotations

from typing import List

import torch


def estimate_exit_costs(num_exits: int) -> List[float]:
    # Use normalized depth as proxy when FLOPs tooling isn't available.
    return [(i + 1) / num_exits for i in range(num_exits)]


def expected_compute(exit_rates: List[float]) -> float:
    costs = estimate_exit_costs(len(exit_rates))
    return sum(c * r for c, r in zip(costs, exit_rates))
