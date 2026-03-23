"""Tests for QuantitativeAnalyzer and StockForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idx_wrapper.forecast import QuantitativeAnalyzer, StockForecaster, _recommend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(prices, volumes=None) -> pd.DataFrame:
    """Create a minimal price DataFrame."""
    data = {"ClosePrice": prices}
    if volumes is not None:
        data["Volume"] = volumes
    return pd.DataFrame(data)


def _trending_up(n: int = 100, start: float = 100.0) -> pd.DataFrame:
    prices = [start + i * 0.5 for i in range(n)]
    volumes = [1_000_000 + i * 1000 for i in range(n)]
    return _make_df(prices, volumes)


def _trending_down(n: int = 100, start: float = 150.0) -> pd.DataFrame:
    prices = [start - i * 0.5 for i in range(n)]
    volumes = [1_000_000 + i * 500 for i in range(n)]
    return _make_df(prices, volumes)


def _flat(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    return _make_df([price] * n)


# ---------------------------------------------------------------------------
# _recommend helper
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_strong_buy(self):
        assert _recommend(0.6) == "Strong Buy"

    def test_buy(self):
        assert _recommend(0.3) == "Buy"

    def test_hold(self):
        assert _recommend(0.0) == "Hold"

    def test_sell(self):
        assert _recommend(-0.3) == "Sell"

    def test_strong_sell(self):
        assert _recommend(-0.6) == "Strong Sell"


# ---------------------------------------------------------------------------
# QuantitativeAnalyzer
# ---------------------------------------------------------------------------


class TestQuantitativeAnalyzerBasic:
    def setup_method(self):
        self.qa = QuantitativeAnalyzer()

    def test_raises_on_insufficient_data(self):
        df = _make_df([100.0] * 10)
        with pytest.raises(ValueError, match="data points"):
            self.qa.analyze(df)

    def test_raises_on_missing_close_column(self):
        df = pd.DataFrame({"Open": [100.0] * 60})
        with pytest.raises(ValueError, match="close-price column"):
            self.qa.analyze(df)

    def test_returns_expected_keys(self):
        df = _trending_up()
        result = self.qa.analyze(df)
        assert "indicators" in result
        assert "short_term_score" in result
        assert "long_term_score" in result
        for key in ("rsi", "macd", "sma_crossover", "bollinger", "momentum", "volume_trend"):
            assert key in result["indicators"]

    def test_scores_in_valid_range(self):
        df = _trending_up()
        result = self.qa.analyze(df)
        assert -1.0 <= result["short_term_score"] <= 1.0
        assert -1.0 <= result["long_term_score"] <= 1.0
        for v in result["indicators"].values():
            assert -1.0 <= v <= 1.0

    def test_uptrend_gives_positive_long_term_score(self):
        df = _trending_up(n=120)
        result = self.qa.analyze(df)
        assert result["long_term_score"] > 0

    def test_downtrend_gives_negative_long_term_score(self):
        df = _trending_down(n=120)
        result = self.qa.analyze(df)
        assert result["long_term_score"] < 0

    def test_accepts_column_alias_close(self):
        data = {"Close": [100.0 + i * 0.5 for i in range(100)]}
        df = pd.DataFrame(data)
        result = self.qa.analyze(df)
        assert "short_term_score" in result

    def test_no_volume_column_sets_volume_trend_zero(self):
        df = _make_df([100.0 + i for i in range(100)])
        result = self.qa.analyze(df)
        assert result["indicators"]["volume_trend"] == 0.0


# ---------------------------------------------------------------------------
# StockForecaster
# ---------------------------------------------------------------------------


class TestStockForecasterInit:
    def test_default_weights(self):
        f = StockForecaster()
        assert f.quant_weight == 0.6
        assert f.sentiment_weight == 0.4

    def test_custom_weights(self):
        f = StockForecaster(quant_weight=0.7, sentiment_weight=0.3)
        assert f.quant_weight == 0.7

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            StockForecaster(quant_weight=0.5, sentiment_weight=0.3)


class TestStockForecasterForecast:
    def setup_method(self):
        self.f = StockForecaster()

    def test_returns_expected_keys(self):
        df = _trending_up()
        result = self.f.forecast(df)
        for key in (
            "recommendation_short_term", "recommendation_long_term",
            "score_short_term", "score_long_term",
            "quant_score_short", "quant_score_long",
            "sentiment_score", "news_sentiment_label",
            "indicators", "weights", "explanation",
        ):
            assert key in result

    def test_no_news_gives_zero_sentiment(self):
        df = _trending_up()
        result = self.f.forecast(df)
        assert result["sentiment_score"] == 0.0

    def test_positive_news_increases_score(self):
        df = _flat()
        result_no_news = self.f.forecast(df)
        result_pos_news = self.f.forecast(
            df,
            news_headlines=["Record high profit and strong growth dividend increase"],
        )
        assert result_pos_news["score_short_term"] >= result_no_news["score_short_term"]

    def test_negative_news_decreases_score(self):
        df = _flat()
        result_no_news = self.f.forecast(df)
        result_neg_news = self.f.forecast(
            df,
            news_headlines=["Bankruptcy fraud massive loss default collapse"],
        )
        assert result_neg_news["score_short_term"] <= result_no_news["score_short_term"]

    def test_scores_clamped(self):
        df = _trending_up()
        result = self.f.forecast(df)
        assert -1.0 <= result["score_short_term"] <= 1.0
        assert -1.0 <= result["score_long_term"] <= 1.0

    def test_recommendation_is_valid_string(self):
        df = _trending_up()
        result = self.f.forecast(df)
        valid = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
        assert result["recommendation_short_term"] in valid
        assert result["recommendation_long_term"] in valid

    def test_uptrend_recommends_buy_or_hold(self):
        df = _trending_up(n=120)
        result = self.f.forecast(df)
        assert result["recommendation_long_term"] in {"Strong Buy", "Buy", "Hold"}

    def test_downtrend_recommends_sell_or_hold(self):
        df = _trending_down(n=120)
        result = self.f.forecast(df)
        assert result["recommendation_long_term"] in {"Strong Sell", "Sell", "Hold"}


# ---------------------------------------------------------------------------
# StockForecaster.backtest
# ---------------------------------------------------------------------------


class TestStockForecasterBacktest:
    def setup_method(self):
        self.f = StockForecaster()

    def _enough_data(self, n: int = 150) -> pd.DataFrame:
        np.random.seed(99)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return _make_df(prices)

    def test_raises_on_insufficient_data(self):
        df = _make_df([100.0] * 40)
        with pytest.raises(ValueError, match="data points"):
            self.f.backtest(df, test_days=30)

    def test_returns_expected_keys(self):
        df = self._enough_data()
        result = self.f.backtest(df, test_days=20)
        for key in (
            "test_days", "total_predictions", "correct_predictions",
            "directional_accuracy", "precision_buy", "precision_sell", "day_by_day",
        ):
            assert key in result

    def test_directional_accuracy_in_range(self):
        df = self._enough_data()
        result = self.f.backtest(df, test_days=20)
        assert 0.0 <= result["directional_accuracy"] <= 1.0

    def test_day_by_day_has_correct_length(self):
        df = self._enough_data()
        result = self.f.backtest(df, test_days=20)
        assert len(result["day_by_day"]) == result["total_predictions"]

    def test_with_news_by_day(self):
        df = self._enough_data()
        news = [["profit growth"] for _ in range(20)]
        result = self.f.backtest(df, test_days=20, news_by_day=news)
        assert result["total_predictions"] > 0
