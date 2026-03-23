"""FastAPI server exposing sample IDX API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from idx_wrapper import IDXClient
from idx_wrapper.forecast import StockForecaster
from idx_wrapper.news import fetch_latest_headlines

app = FastAPI(
    title="IDX Python Wrapper API",
    description=(
        "A Python wrapper for the public Indonesia Stock Exchange (IDX) data "
        "endpoints.  Visit https://www.idx.co.id for the original data source."
    ),
    version="1.0.0",
)

_client = IDXClient()
_forecaster = StockForecaster()

_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IDX Forecast Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
      --bg: #0f0d0d;
      --card: #1a1717;
      --text: #f2f1ef;
      --muted: #a39d96;
      --primary: #f3a046;
      --border: #2b2626;
      --success: #5bbf9c;
      --error: #f06a5c;
      --accent: #f7b469;
      --accent-rgb: 243, 160, 70;
      --primary-hover: #f6b45b;
      --input: #211d1d;
      --input-border: #3a3232;
      --bg-gradient-start: #1b1818;
      --bg-gradient-end: #0b0a0a;
      --pill-bg: #231f1f;
      --pill-border: #3a2f2a;
    }

    body {
      margin: 0;
      background: radial-gradient(
        circle at top,
        var(--bg-gradient-start) 0%,
        var(--bg) 45%,
        var(--bg-gradient-end) 100%
      );
      color: var(--text);
    }

    .container {
      max-width: 960px;
      margin: 32px auto 64px;
      padding: 0 20px;
    }

    h1 {
      margin-bottom: 8px;
      font-size: 28px;
    }

    p {
      line-height: 1.6;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 18px 30px rgba(0, 0, 0, 0.35);
    }

    .grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }

    label {
      display: block;
      font-weight: 600;
      margin-bottom: 6px;
    }

    input, textarea, button {
      font: inherit;
    }

    input, textarea {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--input-border);
      border-radius: 10px;
      background: var(--input);
      color: var(--text);
      box-sizing: border-box;
    }

    input:focus, textarea:focus {
      outline: 2px solid rgba(var(--accent-rgb), 0.35);
      outline-offset: 2px;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(var(--accent-rgb), 0.2);
    }

    textarea {
      min-height: 110px;
      resize: vertical;
    }

    button {
      border: none;
      border-radius: 10px;
      background: var(--primary);
      color: #1a140f;
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 600;
    }

    button.secondary {
      background: #2a2424;
      color: var(--text);
      border: 1px solid var(--border);
    }

    button:hover {
      background: var(--primary-hover);
      color: #1a140f;
      box-shadow: 0 0 0 2px rgba(var(--accent-rgb), 0.18);
    }

    button:disabled {
      opacity: 0.7;
      cursor: not-allowed;
    }

    .muted {
      color: var(--muted);
    }

    .status {
      margin-top: 8px;
      font-weight: 600;
    }

    .status[data-type="success"] {
      color: var(--success);
    }

    .status[data-type="error"] {
      color: var(--error);
    }

    .pill {
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      background: var(--pill-bg);
      border: 1px solid var(--pill-border);
      color: var(--accent);
      font-weight: 600;
      margin-top: 4px;
    }

    .hidden {
      display: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>IDX Forecast Dashboard</h1>
    <p class="muted">
      Pick a stock and generate today’s recommendation using the weighted stacking
      model (60% quantitative indicators + 40% news sentiment).
    </p>

    <div class="card" style="margin-top: 20px;">
      <form id="forecast-form">
        <div class="grid">
          <div>
            <label for="stock-code">Stock code</label>
            <input id="stock-code" list="stock-options" placeholder="e.g. BBCA" required />
            <datalist id="stock-options"></datalist>
            <button type="button" class="secondary" id="load-stocks" style="margin-top: 10px;">
              Load stock list from API
            </button>
            <div id="stock-status" class="status muted" data-type="muted"></div>
          </div>
          <div>
            <label for="history-days">History days</label>
            <input id="history-days" type="number" min="60" value="90" />
            <p class="muted" style="margin-top: 8px;">
              Minimum 60 days of price history are required for indicator coverage.
            </p>
          </div>
        </div>

        <div style="margin-top: 16px;">
          <label for="news-headlines">News headlines (one per line)</label>
          <textarea id="news-headlines" placeholder="Optional: paste today’s headlines"></textarea>
        </div>

        <div style="margin-top: 16px; display: flex; gap: 12px; align-items: center;">
          <button type="submit" id="run-forecast">Run forecast</button>
          <div id="forecast-status" class="status muted" data-type="muted"></div>
        </div>
      </form>
    </div>

    <section id="result-card" class="card hidden" style="margin-top: 20px;">
      <h2 style="margin-top: 0;">Recommendation</h2>
      <div class="grid">
        <div>
          <h3 style="margin-bottom: 6px;">Short-term</h3>
          <div id="rec-short" class="pill">—</div>
          <p class="muted">Score: <span id="score-short">—</span></p>
        </div>
        <div>
          <h3 style="margin-bottom: 6px;">Long-term</h3>
          <div id="rec-long" class="pill">—</div>
          <p class="muted">Score: <span id="score-long">—</span></p>
        </div>
      </div>
      <div style="margin-top: 12px;">
        <p><strong>Sentiment:</strong> <span id="sentiment-label">—</span> (<span id="sentiment-score">—</span>)</p>
        <p><strong>Weights:</strong> <span id="weights">—</span></p>
        <p><strong>As of:</strong> <span id="as-of">—</span></p>
        <p><strong>Explanation:</strong></p>
        <p id="explanation" class="muted">—</p>
      </div>
    </section>
  </div>

  <script>
    (function () {
      const sampleStocks = [
        { code: "BBCA", name: "Bank Central Asia" },
        { code: "BBRI", name: "Bank Rakyat Indonesia" },
        { code: "TLKM", name: "Telkom Indonesia" },
        { code: "ASII", name: "Astra International" },
        { code: "UNVR", name: "Unilever Indonesia" },
      ];

      const datalist = document.getElementById("stock-options");
      const stockInput = document.getElementById("stock-code");
      const loadButton = document.getElementById("load-stocks");
      const stockStatus = document.getElementById("stock-status");
      const forecastStatus = document.getElementById("forecast-status");
      const resultCard = document.getElementById("result-card");
      const runButton = document.getElementById("run-forecast");

      function setStatus(el, message, type) {
        el.textContent = message;
        el.dataset.type = type || "muted";
      }

      function populateList(stocks) {
        datalist.innerHTML = "";
        stocks.forEach((stock) => {
          if (!stock.code) return;
          const option = document.createElement("option");
          option.value = stock.code;
          option.label = stock.name ? `${stock.code} — ${stock.name}` : stock.code;
          datalist.appendChild(option);
        });
      }

      populateList(sampleStocks);

      loadButton.addEventListener("click", async () => {
        setStatus(stockStatus, "Loading stock list from IDX...", "muted");
        try {
          const response = await fetch("/stocks?length=200");
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          const payload = await response.json();
          const rows = Array.isArray(payload.data) ? payload.data : [];
          const stocks = rows
            .map((row) => ({
              code: row.StockCode || row.stockCode || row.code,
              name: row.StockName || row.stockName || row.name || "",
            }))
            .filter((stock) => stock.code);
          if (!stocks.length) {
            throw new Error("No stock data returned");
          }
          populateList(stocks);
          setStatus(stockStatus, `Loaded ${stocks.length} stocks from API.`, "success");
        } catch (err) {
          populateList(sampleStocks);
          setStatus(stockStatus, `Unable to load list (${err.message}). Using sample tickers.`, "error");
        }
      });

      stockInput.addEventListener("input", () => {
        stockInput.value = stockInput.value.toUpperCase();
      });

      document.getElementById("forecast-form").addEventListener("submit", async (event) => {
        event.preventDefault();
        const code = stockInput.value.trim().toUpperCase();
        const historyDays = parseInt(document.getElementById("history-days").value, 10);
        const headlinesRaw = document.getElementById("news-headlines").value;
        const headlines = headlinesRaw
          .split(/\\r?\\n/)
          .map((line) => line.trim())
          .filter(Boolean);

        if (!code) {
          setStatus(forecastStatus, "Please choose a stock code.", "error");
          return;
        }
        if (Number.isNaN(historyDays) || historyDays < 60) {
          setStatus(forecastStatus, "History days must be at least 60.", "error");
          return;
        }

        const payload = { history_days: historyDays };
        if (headlines.length) {
          payload.news_headlines = headlines;
        }

        runButton.disabled = true;
        setStatus(forecastStatus, "Running forecast...", "muted");

        try {
          const response = await fetch(`/forecast/${encodeURIComponent(code)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Forecast request failed.");
          }

          document.getElementById("rec-short").textContent = data.recommendation_short_term || "—";
          document.getElementById("rec-long").textContent = data.recommendation_long_term || "—";
          document.getElementById("score-short").textContent = data.score_short_term ?? "—";
          document.getElementById("score-long").textContent = data.score_long_term ?? "—";
          document.getElementById("sentiment-label").textContent = data.news_sentiment_label || "Neutral";
          document.getElementById("sentiment-score").textContent = data.sentiment_score ?? "0.0";
          const weights = data.weights
            ? `${Math.round(data.weights.quantitative * 100)}% quantitative / ${Math.round(data.weights.sentiment * 100)}% sentiment`
            : "60% quantitative / 40% sentiment";
          document.getElementById("weights").textContent = weights;
          document.getElementById("as-of").textContent = new Date().toLocaleString();
          document.getElementById("explanation").textContent = data.explanation || "—";

          resultCard.classList.remove("hidden");
          setStatus(forecastStatus, `Forecast ready for ${code}.`, "success");
        } catch (err) {
          setStatus(forecastStatus, `Forecast failed: ${err.message}`, "error");
        } finally {
          runButton.disabled = false;
        }
      });
    })();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    """Optional body for the /forecast/{code} endpoint."""

    news_headlines: Optional[List[str]] = None
    history_days: int = 90  # Minimum 60 required for enough indicator data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
def root() -> Dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok", "message": "IDX Python Wrapper is running."}


@app.get("/ui", tags=["UI"], response_class=HTMLResponse)
def ui() -> str:
    """Simple web UI for the forecast endpoint."""
    return _UI_HTML


@app.get("/news", tags=["News"])
def get_news_headlines(
    query: str = Query(..., description="Search query (e.g. BBCA stock)"),
    limit: int = Query(default=5, ge=1, le=50),
    days: int = Query(default=1, ge=1, le=30),
) -> Dict[str, Any]:
    """Fetch recent headlines from Google News RSS for the given query."""
    try:
        headlines = fetch_latest_headlines(query=query, limit=limit, days=days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch headlines: {exc}") from exc

    return {"query": query, "days": days, "headlines": headlines}


@app.get("/stocks", tags=["Stocks"])
def list_stocks(
    code: str = Query(default="", description="Filter by stock ticker code"),
    name: str = Query(default="", description="Filter by company name"),
    start: int = Query(default=0, ge=0, description="Pagination offset"),
    length: int = Query(default=20, ge=1, le=9999, description="Number of records"),
) -> Any:
    """Return trading summary for listed stocks.

    Optionally filter by **code** (ticker) or **name** (company name).

    Example:
    - `/stocks?code=BBCA` – BCA stock summary
    - `/stocks?name=bank&length=5` – first 5 bank stocks
    """
    try:
        return _client.get_stock_summary(
            code=code, name=name, start=start, length=length
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/stocks/{code}", tags=["Stocks"])
def get_stock(code: str) -> Any:
    """Return the latest trading summary for a single stock *code*.

    Example:
    - `/stocks/BBCA` – BCA stock
    - `/stocks/TLKM` – Telkom stock
    """
    try:
        record = _client.get_stock_price(code)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Stock with code '{code.upper()}' not found."
        )
    return record


@app.get("/companies", tags=["Companies"])
def list_companies(
    code: str = Query(default="", description="Filter by stock ticker code"),
    name: str = Query(default="", description="Filter by company name"),
    start: int = Query(default=0, ge=0, description="Pagination offset"),
    length: int = Query(default=20, ge=1, le=9999, description="Number of records"),
) -> Any:
    """Return company profiles for listed IDX companies.

    Example:
    - `/companies?code=BBCA`
    - `/companies?name=astra`
    """
    try:
        return _client.get_company_profiles(
            code=code, name=name, start=start, length=length
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/stocks/{code}/daily", tags=["Stocks"])
def get_stock_daily(
    code: str,
    date_from: Optional[str] = Query(
        default="",
        alias="dateFrom",
        description="Start date in YYYY-MM-DD format",
    ),
    date_to: Optional[str] = Query(
        default="",
        alias="dateTo",
        description="End date in YYYY-MM-DD format",
    ),
    start: int = Query(default=0, ge=0),
    length: int = Query(default=20, ge=1, le=9999),
) -> Any:
    """Return daily OHLCV data for a specific stock.

    Example:
    - `/stocks/BBCA/daily?dateFrom=2024-01-01&dateTo=2024-01-31`
    """
    try:
        return _client.get_stocks_daily(
            code=code.upper(),
            date_from=date_from or "",
            date_to=date_to or "",
            start=start,
            length=length,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/index/{index_id}", tags=["Index"])
def get_index_statistics(
    index_id: str,
    period: str = Query(
        default="daily",
        description="Data frequency: daily, monthly, or yearly",
    ),
    start: int = Query(default=0, ge=0),
    length: int = Query(default=20, ge=1, le=9999),
) -> Any:
    """Return historical statistics for a market index.

    Common index IDs:
    - **COMPOSITE** – Jakarta Composite Index (IHSG)
    - **LQ45** – LQ45 Index
    - **IDX30** – IDX30 Index

    Example:
    - `/index/COMPOSITE?period=monthly`
    - `/index/LQ45`
    """
    try:
        return _client.get_index_statistics(
            index_id=index_id.upper(),
            period=period,
            start=start,
            length=length,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/brokers", tags=["Brokers"])
def list_brokers(
    start: int = Query(default=0, ge=0),
    length: int = Query(default=20, ge=1, le=9999),
) -> Any:
    """Return trading summary per broker member.

    Example:
    - `/brokers?length=10`
    """
    try:
        return _client.get_broker_summary(start=start, length=length)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


@app.post("/forecast/{code}", tags=["Forecast"])
def forecast_stock(
    code: str,
    body: Optional[ForecastRequest] = None,
) -> Any:
    """Generate a **buy / hold / sell** recommendation for a stock by
    combining quantitative technical analysis with news sentiment scoring.

    **Model architecture (weighted stacking):**
    - Quantitative signals (60% weight): RSI, MACD, SMA20/SMA50 crossover,
      Bollinger Bands, price momentum, volume trend.
    - News sentiment (40% weight): keyword-based scoring of supplied
      headlines (-1.0 very negative → +1.0 very positive).

    Final score = 0.6 × quant_score + 0.4 × sentiment_score

    Recommendations: **Strong Buy** (≥0.50), **Buy** (≥0.15),
    **Hold** ([-0.15, 0.15)), **Sell** (≥-0.50), **Strong Sell**.

    **Request body (optional JSON):**
    ```json
    {
      "news_headlines": ["Company reports record profit", "Shares drop on weak guidance"],
      "history_days": 90
    }
    ```
    `history_days` must be ≥ 60 (default 90).

    Example:
    - `POST /forecast/BBCA` — pure quantitative (no news)
    - `POST /forecast/TLKM` with body — quantitative + news sentiment
    """
    req = body or ForecastRequest()
    ticker = code.upper()
    history_days = req.history_days

    if history_days < 60:
        raise HTTPException(
            status_code=422,
            detail="history_days must be at least 60 (need ≥51 records for technical indicators).",
        )

    # Fetch daily price history
    try:
        raw = _client.get_stocks_daily(code=ticker, length=history_days)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch price data: {exc}") from exc

    records = raw.get("data", [])
    if len(records) < 51:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Not enough historical data for '{ticker}' "
                f"(got {len(records)}, need ≥51 days)."
            ),
        )

    price_df = pd.DataFrame(records)
    if "Date" in price_df.columns:
        price_df = price_df.sort_values("Date").reset_index(drop=True)

    try:
        result = _forecaster.forecast(price_df, news_headlines=req.news_headlines)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"code": ticker, **result}
