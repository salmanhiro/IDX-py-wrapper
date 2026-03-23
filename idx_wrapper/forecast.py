"""Stock forecasting engine combining technical analysis and news sentiment."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sentiment import SentimentAnalyzer

# ---------------------------------------------------------------------------
# Weights for individual technical indicators
# ---------------------------------------------------------------------------

# Short-term: favour momentum/oscillators
_SHORT_TERM_WEIGHTS: Dict[str, float] = {
    "rsi": 0.30,
    "macd": 0.25,
    "momentum": 0.25,
    "sma_crossover": 0.10,
    "bollinger": 0.10,
}

# Long-term: favour trend/volume indicators
_LONG_TERM_WEIGHTS: Dict[str, float] = {
    "sma_crossover": 0.35,
    "bollinger": 0.20,
    "volume_trend": 0.20,
    "rsi": 0.15,
    "macd": 0.10,
}

# Recommendation thresholds applied to the final score in [-1, 1]
_THRESHOLDS = [
    (0.50, "Strong Buy"),
    (0.15, "Buy"),
    (-0.15, "Hold"),
    (-0.50, "Sell"),
]


def _recommend(score: float) -> str:
    for threshold, label in _THRESHOLDS:
        if score >= threshold:
            return label
    return "Strong Sell"


# ---------------------------------------------------------------------------
# Quantitative analyser
# ---------------------------------------------------------------------------


class QuantitativeAnalyzer:
    """Compute technical indicators and return normalised scores in [-1, 1].

    Accepts a :class:`pandas.DataFrame` with at minimum a ``ClosePrice``
    column (and optionally ``Volume``).  Column aliases accepted:
    ``close``, ``Close``, ``ClosePrice``.

    Parameters
    ----------
    rsi_period:
        Look-back window for RSI (default 14).
    short_ma:
        Short moving-average window (default 20).
    long_ma:
        Long moving-average window (default 50).
    bb_period:
        Look-back window for Bollinger Bands (default 20).
    momentum_period:
        Look-back window for momentum (default 10).
    macd_fast / macd_slow / macd_signal:
        MACD parameters (default 12 / 26 / 9).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        short_ma: int = 20,
        long_ma: int = 50,
        bb_period: int = 20,
        momentum_period: int = 10,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> None:
        self.rsi_period = rsi_period
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.bb_period = bb_period
        self.momentum_period = momentum_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a dict of individual indicator scores and aggregate scores.

        Parameters
        ----------
        df:
            Historical price DataFrame sorted ascending by date.

        Returns
        -------
        dict
            Keys: ``indicators`` (individual scores), ``short_term_score``,
            ``long_term_score``.
        """
        prices = self._extract_prices(df)
        volumes = self._extract_volumes(df)

        if len(prices) < max(self.long_ma, self.rsi_period + 1):
            raise ValueError(
                f"Need at least {max(self.long_ma, self.rsi_period + 1)} "
                "data points for a reliable analysis."
            )

        indicators: Dict[str, float] = {
            "rsi": self._rsi_score(prices),
            "macd": self._macd_score(prices),
            "sma_crossover": self._sma_crossover_score(prices),
            "bollinger": self._bollinger_score(prices),
            "momentum": self._momentum_score(prices),
            "volume_trend": self._volume_trend_score(volumes) if volumes is not None else 0.0,
        }

        short_score = sum(
            _SHORT_TERM_WEIGHTS.get(k, 0) * v for k, v in indicators.items()
        )
        long_score = sum(
            _LONG_TERM_WEIGHTS.get(k, 0) * v for k, v in indicators.items()
        )

        return {
            "indicators": indicators,
            "short_term_score": max(-1.0, min(1.0, short_score)),
            "long_term_score": max(-1.0, min(1.0, long_score)),
        }

    # ------------------------------------------------------------------
    # Helper: data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prices(df: pd.DataFrame) -> np.ndarray:
        for col in ("ClosePrice", "Close", "close", "AdjClose"):
            if col in df.columns:
                return df[col].astype(float).values
        raise ValueError(
            "DataFrame must contain a close-price column "
            "('ClosePrice', 'Close', or 'close')."
        )

    @staticmethod
    def _extract_volumes(df: pd.DataFrame) -> Optional[np.ndarray]:
        for col in ("Volume", "volume"):
            if col in df.columns:
                return df[col].astype(float).values
        return None

    # ------------------------------------------------------------------
    # Individual indicator scoring functions
    # ------------------------------------------------------------------

    def _rsi_score(self, prices: np.ndarray) -> float:
        """RSI: score +1 when oversold (buy), -1 when overbought (sell)."""
        period = self.rsi_period
        if len(prices) < period + 1:
            return 0.0
        deltas = np.diff(prices[-(period + 1) :])
        gains = deltas[deltas > 0].mean() if (deltas > 0).any() else 0.0
        losses = -deltas[deltas < 0].mean() if (deltas < 0).any() else 1e-9
        rs = gains / max(losses, 1e-9)
        rsi = 100 - 100 / (1 + rs)
        # Map: RSI < 30 → +1 (oversold, buy), RSI > 70 → -1 (overbought, sell)
        if rsi <= 30:
            return 1.0
        if rsi >= 70:
            return -1.0
        # Linear mapping from [30,70] → [+1, -1]
        return 1.0 - (rsi - 30) / 20.0

    def _macd_score(self, prices: np.ndarray) -> float:
        """MACD line vs signal line: positive histogram → bullish."""
        if len(prices) < self.macd_slow + self.macd_signal:
            return 0.0
        series = pd.Series(prices)
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = (macd_line - signal_line).iloc[-1]
        # Normalise by recent price range
        price_range = np.ptp(prices[-self.macd_slow :]) or 1.0
        return max(-1.0, min(1.0, histogram / (price_range * 0.05)))

    def _sma_crossover_score(self, prices: np.ndarray) -> float:
        """SMA20 vs SMA50 crossover: positive gap → bullish trend."""
        if len(prices) < self.long_ma:
            return 0.0
        sma_short = prices[-self.short_ma :].mean()
        sma_long = prices[-self.long_ma :].mean()
        gap = (sma_short - sma_long) / sma_long if sma_long else 0.0
        return max(-1.0, min(1.0, gap * 20))

    def _bollinger_score(self, prices: np.ndarray) -> float:
        """Bollinger Bands: below lower band → +1 (buy), above upper → -1."""
        if len(prices) < self.bb_period:
            return 0.0
        window = prices[-self.bb_period :]
        mid = window.mean()
        std = window.std(ddof=1) or 1.0
        upper = mid + 2 * std
        lower = mid - 2 * std
        last = prices[-1]
        band_range = upper - lower or 1.0
        # Position within bands: 0 at lower, 1 at upper
        pos = (last - lower) / band_range
        # Map [0,1] → [+1,-1]
        return max(-1.0, min(1.0, 1.0 - 2 * pos))

    def _momentum_score(self, prices: np.ndarray) -> float:
        """Price momentum: percentage change over the momentum window."""
        period = self.momentum_period
        if len(prices) < period + 1:
            return 0.0
        pct = (prices[-1] - prices[-(period + 1)]) / (prices[-(period + 1)] if prices[-(period + 1)] != 0 else 1.0)
        return max(-1.0, min(1.0, pct * 10))

    def _volume_trend_score(self, volumes: np.ndarray) -> float:
        """Rising volume over last 10 days vs previous 10 days → bullish."""
        if len(volumes) < 20:
            return 0.0
        recent = volumes[-10:].mean()
        previous = volumes[-20:-10].mean()
        if previous == 0:
            return 0.0
        ratio = recent / previous - 1
        return max(-1.0, min(1.0, ratio * 5))


# ---------------------------------------------------------------------------
# Stacking forecaster
# ---------------------------------------------------------------------------


class StockForecaster:
    """Combine quantitative technical analysis and news sentiment using a
    weighted stacking method to produce short-term and long-term stock
    recommendations.

    Parameters
    ----------
    quant_weight:
        Weight assigned to the quantitative score (default 0.6).
    sentiment_weight:
        Weight assigned to the news sentiment score (default 0.4).
        Must sum to 1.0 with *quant_weight*.

    Example
    -------
    >>> import pandas as pd
    >>> from idx_wrapper.forecast import StockForecaster
    >>> forecaster = StockForecaster()
    >>> df = pd.DataFrame({"ClosePrice": [...], "Volume": [...]})
    >>> result = forecaster.forecast(df, news_headlines=["Profit surges 20%"])
    >>> result["recommendation_short_term"]
    'Buy'
    """

    def __init__(
        self,
        quant_weight: float = 0.6,
        sentiment_weight: float = 0.4,
    ) -> None:
        if not math.isclose(quant_weight + sentiment_weight, 1.0, abs_tol=1e-6):
            raise ValueError("quant_weight + sentiment_weight must equal 1.0")
        self.quant_weight = quant_weight
        self.sentiment_weight = sentiment_weight
        self._quant = QuantitativeAnalyzer()
        self._sentiment = SentimentAnalyzer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forecast(
        self,
        price_df: pd.DataFrame,
        news_headlines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a buy/hold/sell forecast for short and long time horizons.

        Parameters
        ----------
        price_df:
            Historical daily price DataFrame (ascending date order).
            Must contain ``ClosePrice`` (or ``Close`` / ``close``).
        news_headlines:
            Optional list of recent news headline strings.  When not
            supplied the sentiment contribution is treated as neutral (0).

        Returns
        -------
        dict
            ``recommendation_short_term``, ``recommendation_long_term``,
            ``score_short_term``, ``score_long_term``, ``quant_score_short``,
            ``quant_score_long``, ``sentiment_score``, ``indicators``,
            ``news_sentiment_label``, ``explanation``.
        """
        quant = self._quant.analyze(price_df)
        sentiment_score = (
            self._sentiment.aggregate_score(news_headlines)
            if news_headlines
            else 0.0
        )

        q_short = quant["short_term_score"]
        q_long = quant["long_term_score"]

        final_short = self.quant_weight * q_short + self.sentiment_weight * sentiment_score
        final_long = self.quant_weight * q_long + self.sentiment_weight * sentiment_score
        final_short = max(-1.0, min(1.0, final_short))
        final_long = max(-1.0, min(1.0, final_long))

        rec_short = _recommend(final_short)
        rec_long = _recommend(final_long)
        sentiment_label = self._sentiment.label(sentiment_score)

        explanation = self._build_explanation(
            quant["indicators"], sentiment_score, sentiment_label,
            final_short, final_long, news_headlines or [],
        )

        return {
            "recommendation_short_term": rec_short,
            "recommendation_long_term": rec_long,
            "score_short_term": round(final_short, 4),
            "score_long_term": round(final_long, 4),
            "quant_score_short": round(q_short, 4),
            "quant_score_long": round(q_long, 4),
            "sentiment_score": round(sentiment_score, 4),
            "news_sentiment_label": sentiment_label,
            "indicators": {k: round(v, 4) for k, v in quant["indicators"].items()},
            "weights": {
                "quantitative": self.quant_weight,
                "sentiment": self.sentiment_weight,
            },
            "explanation": explanation,
        }

    def backtest(
        self,
        price_df: pd.DataFrame,
        test_days: int = 30,
        news_by_day: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate forecasting accuracy by masking the last *test_days*.

        For each day in the test window the model is trained on all data
        **before** that day and asked to predict whether the next day's
        price will go up or down.  Directional accuracy is then compared
        to the actual movement.

        Parameters
        ----------
        price_df:
            Full historical price DataFrame (ascending date order).
        test_days:
            Number of trailing days to use as the hold-out test set.
        news_by_day:
            Optional list (length = test_days) of news headline lists,
            one per test day.  ``None`` means no news data.

        Returns
        -------
        dict
            Backtest report: ``directional_accuracy``, ``precision_buy``,
            ``precision_sell``, ``total_predictions``, ``day_by_day``
            (list of per-day results).
        """
        min_train = max(self._quant.long_ma, self._quant.rsi_period + 1) + 1
        total = len(price_df)
        if total < min_train + test_days:
            raise ValueError(
                f"Need at least {min_train + test_days} data points for backtesting "
                f"with test_days={test_days}."
            )

        prices = self._quant._extract_prices(price_df)
        day_results = []

        for i in range(test_days):
            train_end = total - test_days + i
            train_df = price_df.iloc[:train_end].copy()
            headlines = (news_by_day[i] if news_by_day and i < len(news_by_day) else None)

            try:
                result = self.forecast(train_df, headlines)
                predicted_direction = "up" if result["score_short_term"] > 0 else "down"
            except ValueError:
                continue

            # Actual direction: compare next day's close to current close
            actual_close = prices[train_end]
            next_close = prices[min(train_end + 1, total - 1)]
            actual_direction = "up" if next_close > actual_close else "down"
            correct = predicted_direction == actual_direction

            day_results.append(
                {
                    "day_index": i,
                    "predicted_direction": predicted_direction,
                    "actual_direction": actual_direction,
                    "correct": correct,
                    "score_short_term": result["score_short_term"],
                    "recommendation": result["recommendation_short_term"],
                    "actual_close": float(actual_close),
                    "next_close": float(next_close),
                }
            )

        if not day_results:
            return {"error": "No valid predictions produced during backtest."}

        correct_count = sum(1 for d in day_results if d["correct"])
        total_preds = len(day_results)
        directional_accuracy = correct_count / total_preds

        # Precision for Buy signals
        buy_preds = [d for d in day_results if d["predicted_direction"] == "up"]
        precision_buy = (
            sum(1 for d in buy_preds if d["correct"]) / len(buy_preds)
            if buy_preds else None
        )

        # Precision for Sell signals
        sell_preds = [d for d in day_results if d["predicted_direction"] == "down"]
        precision_sell = (
            sum(1 for d in sell_preds if d["correct"]) / len(sell_preds)
            if sell_preds else None
        )

        return {
            "test_days": test_days,
            "total_predictions": total_preds,
            "correct_predictions": correct_count,
            "directional_accuracy": round(directional_accuracy, 4),
            "precision_buy": round(precision_buy, 4) if precision_buy is not None else None,
            "precision_sell": round(precision_sell, 4) if precision_sell is not None else None,
            "day_by_day": day_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        indicators: Dict[str, float],
        sentiment_score: float,
        sentiment_label: str,
        final_short: float,
        final_long: float,
        headlines: List[str],
    ) -> str:
        lines = []
        rsi = indicators.get("rsi", 0)
        macd = indicators.get("macd", 0)
        sma = indicators.get("sma_crossover", 0)
        bollinger = indicators.get("bollinger", 0)
        momentum = indicators.get("momentum", 0)
        volume = indicators.get("volume_trend", 0)

        lines.append(
            f"RSI signal is {'bullish (oversold)' if rsi > 0.3 else 'bearish (overbought)' if rsi < -0.3 else 'neutral'}."
        )
        lines.append(
            f"MACD histogram is {'positive (bullish crossover)' if macd > 0 else 'negative (bearish crossover)'}."
        )
        lines.append(
            f"SMA20/SMA50 crossover is {'bullish (short MA above long MA)' if sma > 0 else 'bearish (short MA below long MA)'}."
        )
        lines.append(
            f"Bollinger Band position is {'near lower band (potential bounce)' if bollinger > 0.3 else 'near upper band (potential pullback)' if bollinger < -0.3 else 'within bands'}."
        )
        if volume != 0:
            lines.append(
                f"Volume trend is {'increasing (confirms momentum)' if volume > 0 else 'decreasing (weak conviction)'}."
            )
        if headlines:
            lines.append(
                f"News sentiment is {sentiment_label} ({sentiment_score:+.2f}) "
                f"based on {len(headlines)} headline(s)."
            )
        else:
            lines.append("No news headlines provided; sentiment contribution is neutral (0.0).")
        lines.append(
            f"Combined short-term score: {final_short:+.2f} → {_recommend(final_short)}."
        )
        lines.append(
            f"Combined long-term score: {final_long:+.2f} → {_recommend(final_long)}."
        )
        return " ".join(lines)
