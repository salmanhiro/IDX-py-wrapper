"""FastAPI server exposing sample IDX API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from idx_wrapper import IDXClient
from idx_wrapper.forecast import StockForecaster

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
