"""Backtesting demonstration: masked last-month study.

This script shows the accuracy of the weighted stacking forecasting model
(quantitative + news sentiment) by:

1. Generating one year of realistic synthetic daily price data.
2. Masking the last 30 days as a held-out test set.
3. Running rolling predictions for each day in the test set using only
   data available up to that point (no look-ahead).
4. Comparing predicted price direction to actual movement.
5. Printing a detailed accuracy report and day-by-day results.

Set USE_LIVE_NEWS=1 to fetch recent headlines from Google News RSS and
inject them into the backtest window. This is a best-effort approximation
for "around current time" news; if the fetch fails, the script falls back
to synthetic headlines.

The synthetic prices follow a geometric random walk with a small upward
drift (realistic for a growing emerging-market stock), combined with
occasional sentiment-driven shocks.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from idx_wrapper.forecast import StockForecaster
from idx_wrapper.news import MAX_DAYS_LOOKBACK, fetch_latest_headlines
from idx_wrapper.sentiment import SentimentAnalyzer

# ---------------------------------------------------------------------------
# Reproducible synthetic data generation
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Simulate 365 trading days of BBCA-like price (~9000 IDR range)
N_DAYS = 365
START_PRICE = 9_000.0
DAILY_DRIFT = 0.0003       # ~7% annual drift
DAILY_VOL = 0.012          # ~19% annual volatility

dates = pd.date_range("2023-01-02", periods=N_DAYS, freq="B")  # business days
log_returns = np.random.normal(DAILY_DRIFT, DAILY_VOL, N_DAYS)
prices = START_PRICE * np.exp(np.cumsum(log_returns))
# Add occasional news-driven spikes (±3% on random days)
for _ in range(20):
    idx = random.randint(30, N_DAYS - 1)
    prices[idx:] *= 1 + random.uniform(-0.03, 0.04)

volumes = (
    np.random.lognormal(mean=16.0, sigma=0.4, size=N_DAYS) * 1_000
).astype(int)

df_full = pd.DataFrame(
    {
        "Date": dates,
        "ClosePrice": prices.round(0),
        "Volume": volumes,
    }
)

# ---------------------------------------------------------------------------
# News headlines for each day in the test window
# ---------------------------------------------------------------------------
# By default we simulate a mix of positive and negative headlines. To inject
# live headlines near the current time, set USE_LIVE_NEWS=1 before running.

TEST_DAYS = 30

USE_LIVE_NEWS = os.getenv("USE_LIVE_NEWS", "").lower() in {"1", "true", "yes"}
NEWS_QUERY = os.getenv("NEWS_QUERY", "BBCA stock")
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "5"))

_positive_headlines = [
    "BBCA reports strong quarterly earnings growth",
    "Bank Central Asia dividend increase announced",
    "Indonesian GDP growth beats expectations",
    "BBCA expands digital banking portfolio",
    "Strong foreign inflows into IDX blue chips",
    "Central bank holds rates; banks benefit",
    "BBCA market cap hits record high",
]
_negative_headlines = [
    "Concerns over rising non-performing loans",
    "Foreign investors reduce IDX exposure",
    "Inflation data comes in higher than expected",
    "BBCA profit margins under pressure",
    "Market sell-off amid global risk aversion",
]


def _build_synthetic_news_by_day(prices_window: np.ndarray, test_days: int) -> list[list[str]]:
    news: list[list[str]] = []
    for i in range(test_days):
        price_change = prices_window[i + 1] - prices_window[i]
        if price_change > 0:
            news.append(random.sample(_positive_headlines, k=2))
        else:
            news.append(random.sample(_negative_headlines, k=2))
    return news


prices_test_window = df_full["ClosePrice"].values[-(TEST_DAYS + 1):]
fallback_news_by_day = _build_synthetic_news_by_day(prices_test_window, TEST_DAYS)
news_by_day = fallback_news_by_day
recent_headlines: list[str] | None = None

if USE_LIVE_NEWS:
    print("\nFetching live news headlines for the backtest window...")
    try:
        live_news: list[list[str]] = []
        for day_index in range(TEST_DAYS):
            day_offset = TEST_DAYS - day_index
            days_back = min(MAX_DAYS_LOOKBACK, max(1, day_offset))
            headlines = fetch_latest_headlines(
                NEWS_QUERY,
                limit=NEWS_LIMIT,
                days=days_back,
            )
            live_news.append(headlines)
        if not live_news or not any(live_news):
            raise RuntimeError("No headlines returned from RSS feed.")
        if any(not items for items in live_news):
            print("Some days are missing headlines; filling gaps with synthetic news.")
        news_by_day = [
            items if items else fallback_news_by_day[index]
            for index, items in enumerate(live_news)
        ]
        recent_headlines = next((items for items in reversed(news_by_day) if items), None)
        print(f"Injected live headlines for query: {NEWS_QUERY}")
    except Exception as exc:
        print(f"Live news fetch failed ({exc}); using synthetic headlines instead.")


# ---------------------------------------------------------------------------
# Run backtesting
# ---------------------------------------------------------------------------

forecaster = StockForecaster(quant_weight=0.6, sentiment_weight=0.4)

print("=" * 70)
print("IDX-py-wrapper — Backtesting Report")
print("Model: Quantitative (60%) + News Sentiment (40%)")
print(f"Synthetic data: {N_DAYS} trading days  |  Masked test window: {TEST_DAYS} days")
print("=" * 70)

backtest = forecaster.backtest(df_full, test_days=TEST_DAYS, news_by_day=news_by_day)

print(
    f"\nTotal predictions : {backtest['total_predictions']}"
)
print(f"Correct           : {backtest['correct_predictions']}")
print(f"Directional acc.  : {backtest['directional_accuracy']:.1%}")
if backtest["precision_buy"] is not None:
    print(f"Precision (Buy)   : {backtest['precision_buy']:.1%}")
if backtest["precision_sell"] is not None:
    print(f"Precision (Sell)  : {backtest['precision_sell']:.1%}")

print("\n--- Day-by-day results (last 10 shown) ---")
print(
    f"{'Day':>4}  {'Score':>7}  {'Predicted':>9}  {'Actual':>7}  "
    f"{'Close':>8}  {'Next':>8}  {'OK?':>4}"
)
print("-" * 60)
for row in backtest["day_by_day"][-10:]:
    ok_mark = "✓" if row["correct"] else "✗"
    print(
        f"{row['day_index']:>4}  {row['score_short_term']:>+7.4f}  "
        f"{row['predicted_direction']:>9}  {row['actual_direction']:>7}  "
        f"{row['actual_close']:>8.0f}  {row['next_close']:>8.0f}  {ok_mark:>4}"
    )

# ---------------------------------------------------------------------------
# Trend-capture visualisation (ASCII)
# ---------------------------------------------------------------------------

print("\n--- Actual vs. Predicted Trend (test window) ---")
print("  Each row = 1 test day.  P=Predicted Up(↑)/Down(↓)  A=Actual")
print()

day_results = backtest["day_by_day"]
chunks = [day_results[i : i + 5] for i in range(0, len(day_results), 5)]
for chunk in chunks:
    preds = "".join("↑" if d["predicted_direction"] == "up" else "↓" for d in chunk)
    actuals = "".join("↑" if d["actual_direction"] == "up" else "↓" for d in chunk)
    matches = "".join("✓" if d["correct"] else "✗" for d in chunk)
    print(f"  Pred   : {preds}")
    print(f"  Actual : {actuals}")
    print(f"  Match  : {matches}")
    print()

# ---------------------------------------------------------------------------
# Sample single-day forecast (most recent data)
# ---------------------------------------------------------------------------

analyzer = SentimentAnalyzer()
sample_headlines = recent_headlines or [
    "BBCA Q4 profit surges 18% amid strong loan growth",
    "Bank Central Asia dividend raised by 10%",
]
print("=" * 70)
print("Sample forecast — most recent day (all 365 days available)")
print("News headlines:")
for h in sample_headlines:
    print(f"  • {h}")
    print(f"    Sentiment score: {analyzer.score_headline(h):+.2f}")
print()

forecast = forecaster.forecast(df_full, news_headlines=sample_headlines)
print(f"Short-term recommendation : {forecast['recommendation_short_term']}")
print(f"Long-term  recommendation : {forecast['recommendation_long_term']}")
print(f"Short-term score          : {forecast['score_short_term']:+.4f}")
print(f"Long-term  score          : {forecast['score_long_term']:+.4f}")
print(f"Quant score (short)       : {forecast['quant_score_short']:+.4f}")
print(f"Quant score (long)        : {forecast['quant_score_long']:+.4f}")
print(f"Sentiment score           : {forecast['sentiment_score']:+.4f}  ({forecast['news_sentiment_label']})")
print(f"\nWeights: quant={forecast['weights']['quantitative']:.0%}  sentiment={forecast['weights']['sentiment']:.0%}")
print(f"\nExplanation:\n  {forecast['explanation']}")

print()
print("Technical indicators:")
for name, val in forecast["indicators"].items():
    bar_len = int(abs(val) * 20)
    bar = ("█" * bar_len).ljust(20)
    sign = "+" if val >= 0 else "-"
    print(f"  {name:<16} {sign}{abs(val):.4f}  {bar}")
print()
