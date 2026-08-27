from __future__ import annotations

from ai.models import Prediction
from app.trading.models import Side, TradeProposal


class DecisionEngine:
    """Converts a prediction into a proposal.

    It never executes a trade. The backend risk engine remains the authority.
    """

    def propose(self, prediction: Prediction, entry: float, stop_loss: float, take_profit: float):
        if prediction.long_probability >= max(
            prediction.short_probability,
            prediction.no_trade_probability,
        ):
            side = Side.LONG
            confidence = prediction.long_probability
        elif prediction.short_probability >= prediction.no_trade_probability:
            side = Side.SHORT
            confidence = prediction.short_probability
        else:
            return None

        return TradeProposal(
            symbol=prediction.symbol,
            side=side,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=f"AI model {prediction.model_version}; regime={prediction.regime.value}",
        )
