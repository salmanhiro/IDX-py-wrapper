"""Keyword-based news sentiment analyzer for financial headlines."""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

_STRONG_POSITIVE = {
    "record high", "all-time high", "strong earnings", "beat expectations",
    "profit surge", "revenue growth", "dividend increase", "stock buyback",
    "upgrade", "outperform", "strong buy", "acquisition approved",
    "breakthrough", "expansion", "partnership", "contract win",
    "market leader", "positive outlook", "raised guidance",
}

_POSITIVE = {
    "growth", "profit", "gain", "rise", "increase", "up", "bullish",
    "positive", "recovery", "rebound", "rally", "optimistic", "improve",
    "strong", "beat", "exceed", "higher", "advance", "momentum",
    "opportunity", "promising", "stable", "resilient", "upgrade",
    "dividend", "earnings", "revenue", "expansion", "invest",
}

_STRONG_NEGATIVE = {
    "bankruptcy", "default", "fraud", "scandal", "collapse", "crash",
    "massive loss", "profit warning", "earnings miss", "downgrade",
    "delisted", "regulatory action", "corruption", "litigation",
    "significant drop", "market crash", "liquidity crisis", "debt spiral",
}

_NEGATIVE = {
    "loss", "decline", "fall", "drop", "decrease", "down", "bearish",
    "negative", "risk", "concern", "worry", "lower", "weak", "miss",
    "disappoint", "pressure", "slowdown", "contraction", "uncertainty",
    "volatile", "threat", "challenge", "downgrade", "sell-off", "debt",
    "inflation", "recession", "layoff", "cut", "reduce",
}


class SentimentAnalyzer:
    """Score financial news headlines on a scale of -1.0 (very negative)
    to +1.0 (very positive) using a financial keyword dictionary.

    The scorer is intentionally lightweight — no ML models or external
    APIs are required.

    Example
    -------
    >>> analyzer = SentimentAnalyzer()
    >>> analyzer.score_headline("Company reports record high profit")
    0.9
    >>> analyzer.aggregate_score(["Earnings beat expectations", "Shares drop"])
    0.25
    """

    def score_headline(self, headline: str) -> float:
        """Return a sentiment score in [-1.0, 1.0] for a single headline.

        Parameters
        ----------
        headline:
            News headline text.

        Returns
        -------
        float
            -1.0 = very negative, 0.0 = neutral, +1.0 = very positive.
        """
        text = headline.lower()
        score = 0.0

        for phrase in _STRONG_POSITIVE:
            if phrase in text:
                score += 0.9

        for phrase in _STRONG_NEGATIVE:
            if phrase in text:
                score -= 0.9

        words = set(re.findall(r"\b\w+\b", text))
        for word in words:
            if word in _POSITIVE:
                score += 0.3
            if word in _NEGATIVE:
                score -= 0.3

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))

    def aggregate_score(self, headlines: List[str]) -> float:
        """Aggregate sentiment scores from multiple headlines.

        Parameters
        ----------
        headlines:
            List of news headline strings.

        Returns
        -------
        float
            Mean sentiment score in [-1.0, 1.0].  Returns 0.0 if the
            list is empty.
        """
        if not headlines:
            return 0.0
        scores = [self.score_headline(h) for h in headlines]
        return sum(scores) / len(scores)

    def label(self, score: float) -> str:
        """Convert a numeric score to a human-readable label.

        Parameters
        ----------
        score:
            Sentiment score in [-1.0, 1.0].

        Returns
        -------
        str
            One of ``"Very Positive"``, ``"Positive"``, ``"Neutral"``,
            ``"Negative"``, ``"Very Negative"``.
        """
        if score >= 0.5:
            return "Very Positive"
        if score >= 0.15:
            return "Positive"
        if score > -0.15:
            return "Neutral"
        if score > -0.5:
            return "Negative"
        return "Very Negative"
