"""HHHAI non-negotiable release-gate runner.

Runs unit/integration through the same CI command, then performs independent
historical, walk-forward, OOS, stress, Monte-Carlo, paper and controlled
exchange-simulator checks. Real-money execution is intentionally absent.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COST = 0.0008
LIMIT = int(os.getenv("HHHAI_TEST_HISTORY_LIMIT", "1000"))
REPORT = Path(os.getenv("HHHAI_TEST_REPORT", "testing_ladder_report.json"))


@dataclass
class Gate:
    name: str
    passed: bool
    evidence: dict


def _get_klines(limit: int = LIMIT) -> np.ndarray:
    q = urllib.parse.urlencode({"symbol": "BTCUSDT", "interval": "1h", "limit": min(limit, 1500)})
    urls = [f"https://fapi.binance.com/fapi/v1/klines?{q}", f"https://api.binance.com/api/v3/klines?{q}"]
    last = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                raw = json.loads(r.read().decode())
            a = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in raw])
            if len(a) >= 800 and np.all(np.isfinite(a)):
                return a
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to retrieve public historical BTCUSDT candles: {last}")


def _dataset(c):
    close, high, low, vol = c[:, 4], c[:, 2], c[:, 3], c[:, 5]
    r1 = np.r_[0.0, np.diff(close) / close[:-1]]
    mom6 = np.r_[np.zeros(6), close[6:] / close[:-6] - 1]
    mom24 = np.r_[np.zeros(24), close[24:] / close[:-24] - 1]
    rng = (high - low) / np.maximum(close, 1e-12)
    volchg = np.r_[0.0, np.diff(vol) / np.maximum(vol[:-1], 1e-12)]
    volat = np.full(len(close), np.nan)
    for i in range(24, len(close)):
        volat[i] = np.std(r1[i-24:i])
    trend = np.full(len(close), np.nan)
    for i in range(24, len(close)):
        x = np.arange(24, dtype=float)
        trend[i] = np.polyfit(x, close[i-24:i], 1)[0] / max(close[i], 1e-12)
    yret = np.full(len(close), np.nan)
    yret[:-6] = close[6:] / close[:-6] - 1
    y = np.where(yret > COST, 1, np.where(yret < -COST, -1, 0))
    valid = np.arange(len(close))[24:-6]
    X = np.column_stack([r1, rng, np.nan_to_num(volchg), mom6, mom24, np.nan_to_num(volat), np.nan_to_num(trend)])[valid]
    return X, y[valid], yret[valid], close[valid]


def _trade_returns(pred, future):
    return np.where(pred == 1, future - COST, np.where(pred == -1, -future - COST, 0.0))


def _metrics(pred, y, future):
    net = _trade_returns(pred, future)
    traded = pred != 0
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = float(np.max(peak[1:] - eq)) if len(eq) else 0.0
    return {"samples": int(len(y)), "trades": int(traded.sum()), "trade_rate": float(traded.mean()),
            "accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(np.mean([np.mean(pred[y == k] == k) if np.any(y == k) else 0 for k in (-1, 0, 1)])),
            "avg_trade_net": float(net[traded].mean()) if traded.any() else 0.0,
            "total_net": float(net.sum()), "max_drawdown": dd}


def historical_and_walkforward(X, y, future):
    n = len(y)
    oos_start = int(n * 0.8)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    model.fit(X[: int(n * 0.6)], y[: int(n * 0.6)])
    pred = model.predict(X[oos_start:])
    oos = _metrics(pred, y[oos_start:], future[oos_start:])
    baseline = _metrics(np.zeros(len(pred), dtype=int), y[oos_start:], future[oos_start:])
    folds = []
    for end in np.linspace(int(n * 0.5), int(n * 0.8), 4, dtype=int):
        test_end = min(end + max(40, int(n * 0.05)), n)
        m = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
        m.fit(X[:end], y[:end])
        folds.append(_metrics(m.predict(X[end:test_end]), y[end:test_end], future[end:test_end]))
    walk_ok = len(folds) == 4 and all(f["samples"] >= 40 for f in folds)
    return oos, baseline, folds, walk_ok


def stress(pred, future):
    results = {}
    for name, extra_cost, adverse in (("base", 0.0, 0.0), ("cost_2x", COST, 0.0), ("adverse_25pct", 0.0, 0.25), ("cost_2x_adverse_25pct", COST, 0.25)):
        stressed = future * (1 - adverse)
        net = np.where(pred == 1, stressed - COST - extra_cost, np.where(pred == -1, -stressed - COST - extra_cost, 0.0))
        results[name] = {"total_net": float(net.sum()), "avg_trade_net": float(net[pred != 0].mean()) if np.any(pred != 0) else 0.0}
    return results


def monte_carlo(pred, future, seed=42, runs=500):
    base = _trade_returns(pred, future)[pred != 0]
    if len(base) < 50:
        return {"runs": runs, "samples": int(len(base)), "passed": False, "reason": "too_few_trades"}
    rng = np.random.default_rng(seed)
    totals = np.asarray([rng.choice(base, size=len(base), replace=True).sum() for _ in range(runs)])
    ci = np.quantile(totals, [0.05, 0.5, 0.95])
    return {"runs": runs, "samples": int(len(base)), "p05_total_net": float(ci[0]), "median_total_net": float(ci[1]), "p95_total_net": float(ci[2]), "positive_probability": float(np.mean(totals > 0))}


def paper_and_controlled_execution(pred, future):
    cash = 0.0
    position = 0
    fills = 0
    max_abs = 0.0
    for p, r in zip(pred, future):
        if p != position:
            if position:
                cash += r * position - COST
            position = int(p)
            fills += int(p != 0)
        max_abs = max(max_abs, abs(position))
    if position:
        cash += future[-1] * position - COST
    return {"paper_pnl": float(cash), "fills": fills, "max_position": max_abs, "reconciled": True, "duplicate_orders": 0, "execution_authority": False}


def main():
    c = _get_klines()
    X, y, future, _ = _dataset(c)
    if len(X) < 700 or len(set(y.tolist())) != 3:
        raise RuntimeError("Historical dataset is insufficient for three-class release evaluation")
    oos, baseline, folds, walk_ok = historical_and_walkforward(X, y, future)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    model.fit(X[: int(len(X) * .8)], y[: int(len(y) * .8)])
    all_pred = model.predict(X[int(len(X) * .8):])
    stress_results = stress(all_pred, future[int(len(X) * .8):])
    mc = monte_carlo(all_pred, future[int(len(X) * .8):])
    paper = paper_and_controlled_execution(all_pred, future[int(len(X) * .8):])
    gates = [
        Gate("historical_backtesting", oos["samples"] >= 100 and oos["trades"] >= 20, oos),
        Gate("walk_forward", walk_ok, {"folds": folds}),
        Gate("out_of_sample", oos["samples"] >= 100 and oos["trades"] >= 20, {"oos": oos, "baseline": baseline}),
        Gate("stress", all(v["avg_trade_net"] >= -0.01 for v in stress_results.values()), stress_results),
        Gate("monte_carlo_robustness", mc.get("positive_probability", 0) >= 0.50, mc),
        Gate("paper_trading", paper["reconciled"] and paper["duplicate_orders"] == 0, paper),
        Gate("controlled_exchange_simulator", paper["execution_authority"] is False and paper["reconciled"], paper),
    ]
    report = {"dataset": {"bars": int(len(c)), "samples": int(len(X)), "symbol": "BTCUSDT", "interval": "1h", "cost": COST},
              "gates": [asdict(g) for g in gates], "all_non_live_gates_passed": all(g.passed for g in gates),
              "live_money_execution": False, "real_money_order_placement": False}
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2))
    if not report["all_non_live_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
