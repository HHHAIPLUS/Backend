from datetime import datetime, timezone, timedelta
from app.market_data.realtime import NewsEvent, assess_news

def test_news_cluster_creates_risk_flag():
    now=datetime.now(timezone.utc)
    events=[NewsEvent(source="test",title=str(i),published_at=now-timedelta(minutes=5),sentiment=0,relevance=0,impact=0,credibility=.8) for i in range(5)]
    risk, sentiment, flags=assess_news(events)
    assert risk > 0
    assert "fresh_news_cluster" in flags

def test_old_news_does_not_drive_fresh_cluster():
    old=datetime.now(timezone.utc)-timedelta(hours=3)
    event=NewsEvent(source="test",title="old",published_at=old,sentiment=0,relevance=0,impact=0,credibility=.8)
    risk, _, flags=assess_news([event])
    assert risk == 0
    assert flags == []
