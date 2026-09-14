from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from types import MethodType
from typing import Any

from app.persistence.supabase import store
from app.persistence.repository import record_event
from app.ml.predictive import predictive_model
from app.ml.live_features import enrich_missing_features

log = logging.getLogger("hhhai.multi_coin_selection")


DEFAULT_MAX_ACTIVE_SYMBOLS = 5
DEFAULT_UNIVERSE_REFRESH_SECONDS = 300
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"


def _binance_get(path: str, params: dict[str, str] | None = None) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = f"{BINANCE_FUTURES_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "HHHAI/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _rank_binance_symbols(limit: int) -> list[str]:
    """Select a small liquid Binance USD-M perpetual universe before full AI analysis."""
    exchange_info = _binance_get("/fapi/v1/exchangeInfo")
    eligible = {
        item.get("symbol")
        for item in exchange_info.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    }
    eligible.discard(None)

    tickers = _binance_get("/fapi/v1/ticker/24hr")
    ranked: list[tuple[float, str]] = []
    for ticker in tickers:
        symbol = ticker.get("symbol")
        if symbol not in eligible:
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            trade_count = float(ticker.get("count") or 0.0)
            high = float(ticker.get("highPrice") or 0.0)
            low = float(ticker.get("lowPrice") or 0.0)
            last = float(ticker.get("lastPrice") or 0.0)
            if quote_volume <= 0 or last <= 0:
                continue
            range_pct = max(0.0, (high - low) / last)
            # Liquidity is primary; trade activity and usable volatility break ties.
            score = math.log1p(quote_volume) * 0.70 + math.log1p(trade_count) * 0.20 + min(range_pct, 1.0) * 10.0 * 0.10
            ranked.append((score, symbol))
        except (TypeError, ValueError):
            continue

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [symbol for _, symbol in ranked[:limit]]


def _dynamic_symbols(limit: int) -> list[str]:
    """Return dynamic symbols, with an explicit env override available for controlled tests."""
    explicit = os.getenv("HHHAI_TRADE_SYMBOLS", "").strip()
    if explicit:
        symbols = [x.strip().upper() for x in explicit.split(",") if x.strip()]
        return symbols[:limit]
    try:
        symbols = _rank_binance_symbols(limit)
        if symbols:
            return symbols
    except Exception as exc:
        log.warning("Dynamic Binance universe selection failed: %s", exc)
    return ["BTCUSDT"]


def install_multi_coin_selection(trader: Any) -> None:
    """Install dynamic portfolio selection without enabling execution."""
    cls = trader.__class__
    if getattr(cls, "_hhhai_multi_coin_installed", False):
        return

    original_execute = cls._execute
    original_risk_check = cls._risk_check
    original_predict = predictive_model.predict

    def enriched_predict(features):
        try:
            symbol = os.getenv("HHHAI_LIVE_FEATURE_SYMBOL", "BTCUSDT")
            enriched = enrich_missing_features(symbol, dict(features or {}))
            return original_predict(enriched)
        except Exception as exc:
            log.warning("Live predictive feature enrichment failed: %s", exc)
            return original_predict(features)

    predictive_model.predict = enriched_predict

    async def guarded_risk(self, world, decision, candidate):
        result = await original_risk_check(self, world, decision, candidate)
        if result.get("allowed") and candidate:
            equity = float(result.get("equity") or 0.0)
            quantity = float(result.get("quantity") or 0.0)
            entry = float(candidate.entry or 0.0)
            true_leverage = (quantity * entry / equity) if equity > 0 else float("inf")
            if true_leverage > 5.0 + 1e-9:
                reasons = list(result.get("reasons") or [])
                reasons.append(f"true leverage {true_leverage:.2f}x exceeds 5x maximum")
                result = {**result, "allowed": False, "decision": "block", "reasons": reasons, "leverage": true_leverage}
            else:
                result = {**result, "leverage": true_leverage}
        return result

    async def portfolio_run(self):
        max_active_symbols = max(1, int(os.getenv("HHHAI_MAX_ACTIVE_SYMBOLS", str(DEFAULT_MAX_ACTIVE_SYMBOLS))))
        refresh_seconds = max(60, int(os.getenv("HHHAI_UNIVERSE_REFRESH_SECONDS", str(DEFAULT_UNIVERSE_REFRESH_SECONDS))))
        last_universe_refresh = 0.0
        next_decision = 0.0
        next_management = 0.0

        while self.running:
            now = asyncio.get_running_loop().time()

            if now >= last_universe_refresh:
                selected = await asyncio.to_thread(_dynamic_symbols, max_active_symbols)
                self.config.symbols = tuple(selected[:max_active_symbols]) or ("BTCUSDT",)
                last_universe_refresh = now + refresh_seconds
                log.info("HHHAI dynamic universe selected: %s", self.config.symbols)

            if self.execution_mode in {"testnet", "live"} and now >= next_management:
                reviews = 0
                for symbol in self.config.symbols:
                    if reviews >= self.max_position_reviews_per_cycle:
                        break
                    try:
                        managed = await self._manage_open_positions(symbol)
                        reviews += len(managed)
                        self.last_error = None
                    except Exception as exc:
                        self.last_error = f"{type(exc).__name__}: {exc}"
                        log.exception("Position-management cycle failed for %s", symbol)
                next_management = now + self.position_review_interval

            if now >= next_decision:
                pending: dict[str, tuple] = {}
                scan_results: list[dict[str, Any]] = []

                async def defer_execute(this, symbol, world, decision, candidate, risk):
                    pending[symbol] = (symbol, world, decision.copy(), candidate, dict(risk))
                    return {"status": "selection_pending", "reason": "portfolio selector is ranking selected symbols before execution", "live_exchange_order": False}

                self._execute = MethodType(defer_execute, self)
                try:
                    for symbol in self.config.symbols:
                        previous_feature_symbol = os.environ.get("HHHAI_LIVE_FEATURE_SYMBOL")
                        os.environ["HHHAI_LIVE_FEATURE_SYMBOL"] = symbol
                        try:
                            result = await self.run_cycle(symbol)
                            scan_results.append(result)
                            self.last_error = None
                        except Exception as exc:
                            self.last_error = f"{type(exc).__name__}: {exc}"
                            log.exception("Autonomous decision cycle failed for %s", symbol)
                        finally:
                            if previous_feature_symbol is None:
                                os.environ.pop("HHHAI_LIVE_FEATURE_SYMBOL", None)
                            else:
                                os.environ["HHHAI_LIVE_FEATURE_SYMBOL"] = previous_feature_symbol
                finally:
                    self._execute = MethodType(original_execute, self)

                qualified = []
                for result in scan_results:
                    decision = result.get("decision") or {}
                    risk = result.get("risk") or {}
                    optimizer = result.get("optimizer") or {}
                    symbol = result.get("symbol")
                    if not symbol or symbol not in pending:
                        continue
                    if decision.get("action") not in {"LONG", "SHORT"}:
                        continue
                    if not decision.get("execution_candidate") or not risk.get("allowed"):
                        continue
                    if optimizer.get("decision") != "trade":
                        continue
                    candidate = pending[symbol][3]
                    entry = max(float(candidate.entry or 0), 1e-12)
                    normalized_ev = float(optimizer.get("expected_value") or 0.0) / entry
                    confidence = float(decision.get("confidence") or 0.0)
                    quality = float(optimizer.get("trade_quality") or 0.0)
                    timing = float(optimizer.get("entry_timing") or 0.0)
                    score = 0.40 * normalized_ev + 0.25 * quality + 0.20 * confidence + 0.15 * timing
                    qualified.append((score, symbol, result))

                qualified.sort(key=lambda x: x[0], reverse=True)
                winner = qualified[0] if qualified else None
                actual_execution = {"status": "not_executed", "reason": "no qualified portfolio-level opportunity", "live_exchange_order": False}

                if winner:
                    _, winner_symbol, winner_result = winner
                    args = pending[winner_symbol]
                    try:
                        actual_execution = await original_execute(self, *args)
                        winner_result["execution"] = actual_execution
                        winner_result["portfolio_selection"] = {"selected": winner_symbol, "score": winner[0], "candidates_considered": len(scan_results), "qualified_candidates": len(qualified)}
                        self.last_cycle = winner_result
                        if store.configured:
                            try:
                                await record_event("portfolio_execution", {"selected_symbol": winner_symbol, "score": winner[0], "candidates_considered": len(scan_results), "qualified_candidates": len(qualified), "execution": actual_execution, "created_at": datetime.now(timezone.utc).isoformat()})
                            except Exception:
                                log.exception("Failed to persist portfolio execution event")
                    except Exception as exc:
                        self.last_error = f"{type(exc).__name__}: {exc}"
                        log.exception("Selected opportunity execution failed for %s", winner_symbol)
                else:
                    self.last_cycle = {"status": "no_trade", "symbols_scanned": list(self.config.symbols), "candidates_considered": len(scan_results), "qualified_candidates": 0, "results": scan_results}

                self.last_cycle_at = datetime.now(timezone.utc)
                next_decision = now + self.config.interval_seconds

            try:
                await asyncio.sleep(min(1.0, self.position_review_interval))
            except asyncio.CancelledError:
                break

        log.info("Autonomous trader stopped")

    cls._risk_check = guarded_risk
    cls._run = portfolio_run
    cls._hhhai_multi_coin_installed = True
    log.info("Installed HHHAI dynamic top-5 portfolio selection, true-leverage guard, and predictive candle enrichment")
