from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx

from app.market_data.realtime import NewsEvent


POSITIVE = {"surge", "rally", "approval", "approved", "inflow", "adoption", "bullish", "breakout", "partnership", "launch", "growth", "record"}
NEGATIVE = {"hack", "exploit", "lawsuit", "ban", "banned", "outflow", "liquidation", "liquidations", "fraud", "crash", "bearish", "investigation", "attack", "shutdown", "default", "delist", "delisting"}
HIGH_IMPACT = {"hack", "exploit", "ban", "lawsuit", "fraud", "shutdown", "default", "liquidation", "delisting", "approval", "approved"}


@dataclass(frozen=True)
class SourcePolicy:
    name: str
    weight: float


DEFAULT_SOURCES = {
    "coindesk": SourcePolicy("coindesk", 0.90),
    "cointelegraph": SourcePolicy("cointelegraph", 0.80),
    "binance": SourcePolicy("binance", 1.00),
    "bitget": SourcePolicy("bitget", 1.00),
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def classify_text(title: str, description: str = "") -> tuple[float, float, float, list[str]]:
    tokens = _tokenize(f"{title} {description}")
    pos = sum(t in POSITIVE for t in tokens)
    neg = sum(t in NEGATIVE for t in tokens)
    impact_hits = [t for t in tokens if t in HIGH_IMPACT]
    total = pos + neg
    sentiment = (pos - neg) / total if total else 0.0
    impact = min(1.0, 0.25 + 0.15 * len(impact_hits) + 0.05 * min(5, total))
    relevance = min(1.0, 0.2 + 0.1 * sum(t in {"bitcoin", "btc", "ethereum", "eth", "crypto", "futures", "binance", "bitget", "market"} for t in tokens))
    return max(-1.0, min(1.0, sentiment)), relevance, impact, sorted(set(impact_hits))


def parse_rss(xml: str, source: str, *, fetched_at: datetime | None = None) -> list[NewsEvent]:
    fetched_at = fetched_at or datetime.now(timezone.utc)
    root = ET.fromstring(xml)
    events: list[NewsEvent] = []
    policy = DEFAULT_SOURCES.get(source.lower(), SourcePolicy(source.lower(), 0.50))
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        description = (item.findtext("description") or "").strip()
        raw_date = (item.findtext("pubDate") or "").strip()
        try:
            published = parsedate_to_datetime(raw_date).astimezone(timezone.utc) if raw_date else fetched_at
        except (TypeError, ValueError):
            published = fetched_at
        sentiment, relevance, impact, keywords = classify_text(title, description)
        events.append(NewsEvent(source=source, title=title, url=(item.findtext("link") or None), published_at=published, sentiment=sentiment, relevance=relevance, impact=impact, credibility=policy.weight, keywords=keywords))
    return events


def fetch_rss(url: str, source: str, timeout: float = 10.0) -> list[NewsEvent]:
    with httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False, headers={"User-Agent": "HHHAI/2.0"}) as client:
        response = client.get(url)
        response.raise_for_status()
        return parse_rss(response.text, source)


def aggregate_news(events: list[NewsEvent], now: datetime | None = None, horizon_hours: int = 24) -> dict[str, float | int]:
    now = now or datetime.now(timezone.utc)
    active = [e for e in events if 0 <= (now - e.published_at).total_seconds() <= horizon_hours * 3600]
    if not active:
        return {"count": 0, "sentiment": 0.0, "risk": 0.0, "credibility": 0.0}
    weights = [max(0.01, e.relevance * e.credibility) for e in active]
    total = sum(weights)
    sentiment = sum(e.sentiment * w for e, w in zip(active, weights)) / total
    risk = min(1.0, sum(e.impact * e.relevance * e.credibility for e in active) / max(1.0, len(active)))
    credibility = sum(e.credibility * e.relevance for e in active) / total
    return {"count": len(active), "sentiment": max(-1.0, min(1.0, sentiment)), "risk": max(0.0, min(1.0, risk)), "credibility": max(0.0, min(1.0, credibility))}
