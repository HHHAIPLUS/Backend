from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(candidate_returns: np.ndarray, baseline_returns: np.ndarray, seed: int = 42, samples: int = 2000, alpha: float = 0.05) -> dict[str, float | bool]:
    candidate = np.asarray(candidate_returns, dtype=float)
    baseline = np.asarray(baseline_returns, dtype=float)
    if candidate.shape != baseline.shape or candidate.size < 100:
        return {"valid": False, "difference_mean": 0.0, "ci_low": float("nan"), "ci_high": float("nan")}
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        return {"valid": False, "difference_mean": 0.0, "ci_low": float("nan"), "ci_high": float("nan")}
    diff = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(samples, diff.size))
    means = diff[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"valid": True, "difference_mean": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "alpha": float(alpha), "samples": int(samples)}


def promotion_gate(candidate_returns: np.ndarray, baseline_returns: np.ndarray, candidate_balanced_accuracy: float, baseline_balanced_accuracy: float, candidate_drawdown: float, baseline_drawdown: float, min_samples: int = 100) -> dict[str, object]:
    """Conservative economic + statistical promotion gate on untouched paired OOS observations."""
    ci = paired_bootstrap_ci(candidate_returns, baseline_returns)
    drawdown_ok = candidate_drawdown <= max(1e-12, baseline_drawdown * 1.25) if baseline_drawdown > 0 else candidate_drawdown <= 1e-12
    accuracy_ok = candidate_balanced_accuracy >= baseline_balanced_accuracy
    statistically_better = bool(ci.get("valid") and float(ci["ci_low"]) > 0.0)
    enough = len(candidate_returns) >= min_samples
    promoted = bool(enough and accuracy_ok and drawdown_ok and statistically_better)
    return {
        "promoted": promoted,
        "enough_samples": enough,
        "balanced_accuracy_not_worse": accuracy_ok,
        "drawdown_within_limit": drawdown_ok,
        "statistically_positive": statistically_better,
        "paired_bootstrap": ci,
    }
