from __future__ import annotations

from ai.models import NewsSignal


class NewsIntelligence:
    """Interface for Phase 2 news processing.

    External news providers are intentionally not hard-coded yet. Provider
    selection, source validation, deduplication and scoring will be finalized
    before production use.
    """

    def score(self, signal: NewsSignal) -> float:
        return signal.impact * signal.credibility * signal.relevance
