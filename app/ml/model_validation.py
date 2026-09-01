from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(candidate_returns: np.ndarray, baseline_returns: np.ndarray, seed: int = 42, samples: int = 2000, alpha: float = 0.05) -> dict[str, float | bool]:
    candidate = np.asarray(candidate_returns, dtype=float)
    baseline = np.asarray(baseline_returns, dtype=float)
    if candidate.shape != baseline.shape or candidate.size < 100:
        return {"valid": False, "difference_mean": 0.0, "ci_low": float("nan"), "ci_high": float("nan")}
    diff = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(samples, diff.size))
    means = diff[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"valid": True, "difference_mean": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "alpha": float(alpha), "samples": int(samples)}
