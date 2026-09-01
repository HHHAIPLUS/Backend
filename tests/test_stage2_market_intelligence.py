from datetime import datetime, timedelta, timezone

from app.market_data.cache import MarketDataCache
from app.market_data.intelligence import MarketBar, PointInTimeContext, aggregate_timeframes, build_market_state, canonical_model_features, point_in_time_join
from app.market_data.news import aggregate_news, classify_text, parse_rss
from app.market_data.realtime import NewsEvent


def bars(n=60, start=None, step=0.001):
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [MarketBar(symbol="BTCUSDT", timeframe="5m", timestamp=start + timedelta(minutes=5 * i), open=100 + i * step, high=100.2 + i * step, low=99.8 + i * step, close=100 + (i + 1) * step, volume=1000 + i * 5, quote_volume=100000 + i * 500) for i in range(n)]


def test_point_in_time_join_never_uses_future_context():
    b = bars(3)
    context = [PointInTimeContext(timestamp=b[1].timestamp, funding_rate=0.01), PointInTimeContext(timestamp=b[2].timestamp + timedelta(seconds=1), funding_rate=0.99)]
    joined = point_in_time_join(b, context, max_age_seconds=600)
    assert joined[0][1] is None
    assert joined[1][1].funding_rate == 0.01
    assert joined[2][1].funding_rate == 0.01


def test_timeframe_aggregation_preserves_ohlcv_without_lookahead():
    source = bars(12)
    result = aggregate_timeframes(source, 15)
    assert result
    assert result[0].open == source[0].open
    assert result[0].close == source[2].close
    assert result[0].high == max(x.high for x in source[:3])
    assert result[0].volume == sum(x.volume for x in source[:3])


def test_market_state_contains_structure_derivatives_regime_and_quality():
    b = bars(60)
    context = PointInTimeContext(timestamp=b[-1].timestamp, funding_rate=0.0001, open_interest=1234, order_book_imbalance=0.25, spread_bps=1.2)
    state = build_market_state("BTCUSDT", {"5m": b}, context, observed_at=b[-1].timestamp)
    assert state.symbol == "BTCUSDT"
    assert state.timeframes["5m"]["close"] is not None
    assert "trend" in state.price_structure
    assert state.derivatives["open_interest"] == 1234
    assert state.data_quality >= 0.67
    assert state.usable is True


def test_canonical_model_features_have_stable_schema():
    b = bars(60)
    context = PointInTimeContext(timestamp=b[-1].timestamp, funding_rate=0.0001, open_interest=1234, order_book_imbalance=0.25, spread_bps=1.2)
    state = build_market_state("BTCUSDT", {"5m": b}, context, observed_at=b[-1].timestamp)
    features = canonical_model_features(state)
    expected = {"return_1", "range_pct", "volume_change", "order_book_imbalance", "funding_rate", "open_interest_change", "news_risk", "news_sentiment", "volatility_proxy", "trend_strength", "momentum", "liquidity_stress"}
    assert set(features) == expected
    assert all(isinstance(v, float) for v in features.values())


def test_cross_asset_relationships_are_time_aligned():
    primary = bars(60, step=0.01)
    correlated = bars(60, step=0.01)
    correlated = [x.model_copy(update={"symbol": "ETHUSDT"}) for x in correlated]
    context = PointInTimeContext(timestamp=primary[-1].timestamp, funding_rate=0.0001, open_interest=1234, order_book_imbalance=0.25, spread_bps=1.2)
    state = build_market_state("BTCUSDT", {"5m": primary}, context, correlation_bars={"ETHUSDT": correlated}, observed_at=primary[-1].timestamp)
    assert state.correlations["ETHUSDT"] is not None
    assert state.correlations["ETHUSDT"] > 0.9


def test_market_state_fails_quality_when_core_context_missing():
    b = bars(60)
    state = build_market_state("BTCUSDT", {"5m": b}, None, observed_at=b[-1].timestamp)
    assert state.data_quality < 0.67
    assert state.usable is False
    assert "funding_rate" in state.missing_fields


def test_news_classifier_and_aggregation_are_bounded():
    sentiment, relevance, impact, keywords = classify_text("Bitcoin rally after approval, but exploit warning emerges")
    assert -1 <= sentiment <= 1
    assert 0 <= relevance <= 1
    assert 0 <= impact <= 1
    assert keywords
    now = datetime.now(timezone.utc)
    events = [NewsEvent(source="test", title="x", published_at=now, sentiment=-0.5, relevance=1, impact=1, credibility=0.8)]
    aggregate = aggregate_news(events, now=now)
    assert aggregate["count"] == 1
    assert aggregate["risk"] > 0


def test_rss_parser_preserves_published_timestamp_and_provenance():
    xml = """<rss><channel><item><title>Bitcoin rally</title><link>https://example.test/a</link><pubDate>Thu, 01 Jan 2026 00:00:00 GMT</pubDate><description>approval</description></item></channel></rss>"""
    events = parse_rss(xml, "test")
    assert len(events) == 1
    assert events[0].source == "test"
    assert events[0].published_at.tzinfo is not None


def test_cache_round_trip_and_invalidation(tmp_path):
    cache = MarketDataCache(tmp_path / "cache", ttl_seconds=60)
    cache.set("BTCUSDT:5m:2026-01-01", {"close": 123})
    assert cache.get("BTCUSDT:5m:2026-01-01") == {"close": 123}
    cache.invalidate("BTCUSDT:5m:2026-01-01")
    assert cache.get("BTCUSDT:5m:2026-01-01") is None
