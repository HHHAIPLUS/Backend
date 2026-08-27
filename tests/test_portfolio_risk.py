from ai.portfolio_risk import Exposure, PortfolioRiskEngine


def test_empty_portfolio_is_safe():
    result = PortfolioRiskEngine().evaluate(10000, [])
    assert result["decision"] == "allow"


def test_gross_exposure_limit_blocks():
    positions = [
        Exposure("BTCUSDT", "long", 20000),
        Exposure("ETHUSDT", "long", 20000),
    ]
    result = PortfolioRiskEngine().evaluate(10000, positions)
    assert result["decision"] == "block"


def test_correlated_cluster_blocks():
    positions = [
        Exposure("BTCUSDT", "long", 6000, beta_to_btc=1.0),
        Exposure("SOLUSDT", "long", 5000, beta_to_btc=0.9),
    ]
    result = PortfolioRiskEngine().evaluate(10000, positions)
    assert result["decision"] == "block"
    assert any("correlated" in r.lower() for r in result["reasons"])


def test_execution_authority_is_never_granted():
    result = PortfolioRiskEngine().evaluate(
        10000, [Exposure("BTCUSDT", "long", 1000)]
    )
    assert result["execution_authority"] is False
