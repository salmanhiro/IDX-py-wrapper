# IDX-py-wrapper

A Python wrapper for the public [Indonesia Stock Exchange (IDX)](https://www.idx.co.id/en/products/idx-data-services/) data APIs, exposed as a self-hosted REST service that can run locally or inside a Docker container.

---

## Features

| Endpoint | Description |
|---|---|
| `GET /stocks` | Trading summary for all (or filtered) listed stocks |
| `GET /stocks/{code}` | Latest trading summary for a single stock ticker |
| `GET /stocks/{code}/daily` | Daily OHLCV price history for a stock |
| `GET /companies` | Company profiles for listed IDX companies |
| `GET /index/{index_id}` | Historical statistics for a market index (e.g. IHSG) |
| `GET /brokers` | Trading summary per broker member |
| `POST /forecast/{code}` | Buy/Hold/Sell recommendation (quantitative + news sentiment) |
| `GET /ui` | Web dashboard to run forecast recommendations |
| `GET /news` | Latest news headlines for a query (RSS feed) |

Interactive API documentation is available at **`/docs`** (Swagger UI) once the server is running.

---

## Forecast model architecture

The `/forecast/{code}` endpoint (and `StockForecaster` library class) uses a **weighted stacking** approach:

```
Final Score = 0.60 × Quant Score  +  0.40 × Sentiment Score
```

### Quantitative analysis (60% weight)

Six technical indicators are computed on historical OHLCV data, each normalised to [-1, +1]:

| Indicator | Short-term weight | Long-term weight |
|---|---|---|
| RSI (14-day) | 30% | 15% |
| MACD (12/26/9) | 25% | 10% |
| Price Momentum (10-day) | 25% | — |
| SMA20/SMA50 crossover | 10% | 35% |
| Bollinger Bands (20-day) | 10% | 20% |
| Volume trend | — | 20% |

### News sentiment analysis (40% weight)

A keyword-based financial dictionary scores each headline from -1.0 (very negative) to +1.0 (very positive).  Strong-signal phrases (e.g. *"record high"*, *"bankruptcy"*) carry higher weight than single keywords.  Scores are averaged across all supplied headlines.

### Recommendations

| Score range | Label |
|---|---|
| ≥ 0.50 | Strong Buy |
| 0.15 – 0.50 | Buy |
| -0.15 – 0.15 | Hold |
| -0.50 – -0.15 | Sell |
| < -0.50 | Strong Sell |

---

## Quick start

### Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> in your browser for the API, or
<http://localhost:8000/ui> for the forecast dashboard.

### Run with Docker

```bash
# Build and start the container
docker compose up --build
```

The API will be available at <http://localhost:8000>.

---

## Example requests

```bash
# Health check
curl http://localhost:8000/

# Forecast dashboard
# http://localhost:8000/ui

# Latest price for BCA (BBCA)
curl http://localhost:8000/stocks/BBCA

# First 5 bank stocks
curl "http://localhost:8000/stocks?name=bank&length=5"

# Daily OHLCV for BBCA in January 2024
curl "http://localhost:8000/stocks/BBCA/daily?dateFrom=2024-01-01&dateTo=2024-01-31"

# IHSG (Composite) index – last 20 monthly data points
curl "http://localhost:8000/index/COMPOSITE?period=monthly"

# Company profile for Telkom Indonesia
curl "http://localhost:8000/companies?code=TLKM"

# Top 10 brokers
curl "http://localhost:8000/brokers?length=10"

# Latest headlines for BBCA
curl "http://localhost:8000/news?query=BBCA%20stock&days=1&limit=5"

# Forecast for BBCA — pure quantitative (no news)
curl -X POST http://localhost:8000/forecast/BBCA

# Forecast for BBCA — quantitative + news sentiment
curl -X POST http://localhost:8000/forecast/BBCA \
  -H "Content-Type: application/json" \
  -d '{"news_headlines": ["BBCA Q4 profit surges 18%", "Dividend raised by 10%"]}'
```

---

## Python library usage

You can also use `IDXClient` directly in your own scripts:

```python
from idx_wrapper import IDXClient

client = IDXClient()

# Latest price for a single stock
bbca = client.get_stock_price("BBCA")
print(bbca)

# First 5 stocks from the full summary list
summary = client.get_stock_summary(length=5)
for stock in summary["data"]:
    print(stock)

# Daily prices for BBCA — January 2024
daily = client.get_stocks_daily(
    code="BBCA",
    date_from="2024-01-01",
    date_to="2024-01-31",
)
for row in daily["data"]:
    print(row)

# IHSG monthly index statistics
ihsg = client.get_index_statistics(index_id="COMPOSITE", period="monthly", length=5)
for row in ihsg["data"]:
    print(row)
```

### Forecasting (quantitative + sentiment stacking)

```python
import pandas as pd
from idx_wrapper.forecast import StockForecaster
from idx_wrapper.news import fetch_latest_headlines

forecaster = StockForecaster(quant_weight=0.6, sentiment_weight=0.4)

# Build a price DataFrame from IDX daily data
client = IDXClient()
raw = client.get_stocks_daily(code="BBCA", length=90)
price_df = pd.DataFrame(raw["data"]).sort_values("Date").reset_index(drop=True)

# Optional: supply today's news headlines for sentiment scoring
headlines = [
    "BBCA Q4 profit surges 18% amid strong loan growth",
    "Bank Central Asia dividend raised by 10%",
]

# Or fetch the latest headlines at runtime
# headlines = fetch_latest_headlines("BBCA stock", limit=5, days=1)

result = forecaster.forecast(price_df, news_headlines=headlines)
print(result["recommendation_short_term"])   # e.g. "Buy"
print(result["recommendation_long_term"])    # e.g. "Hold"
print(result["score_short_term"])            # e.g. 0.42
print(result["explanation"])
```

### Backtesting — masked last-month accuracy study

```python
# See backtest_example.py for a full walkthrough.
# Quick version:
backtest = forecaster.backtest(price_df, test_days=30)
print(f"Directional accuracy: {backtest['directional_accuracy']:.1%}")
print(f"Total predictions: {backtest['total_predictions']}")
```

Run the included backtesting demo:

```bash
python backtest_example.py
```

To inject live headlines for the test window (best-effort RSS fetch):

```bash
USE_LIVE_NEWS=1 NEWS_QUERY="BBCA stock" NEWS_LIMIT=5 python backtest_example.py
```

Run the included sample script for a full demonstration:

```bash
python example.py
```

---

## Running tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Project structure

```
IDX-py-wrapper/
├── idx_wrapper/
│   ├── __init__.py      # Package entry point
│   ├── client.py        # IDXClient — all API calls
│   ├── forecast.py      # QuantitativeAnalyzer + StockForecaster (weighted stacking)
│   ├── news.py          # RSS-based headline fetcher (Google News)
│   └── sentiment.py     # SentimentAnalyzer — keyword-based news scoring
├── tests/
│   ├── test_client.py   # Unit tests for IDXClient
│   ├── test_app.py      # Unit tests for FastAPI endpoints
│   ├── test_forecast.py # Unit tests for forecasting engine
│   ├── test_news.py     # Unit tests for RSS headline fetcher
│   └── test_sentiment.py# Unit tests for sentiment analyzer
├── app.py               # FastAPI server with sample endpoints
├── example.py           # Standalone usage examples
├── backtest_example.py  # Backtesting demo with masked last-month study
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Data source

All data is fetched from the public IDX APIs at **https://www.idx.co.id**.  
See [IDX Data Services](https://www.idx.co.id/en/products/idx-data-services/) for details.
