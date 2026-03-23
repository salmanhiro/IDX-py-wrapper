"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STOCK_DATA = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [
        {
            "StockCode": "BBCA",
            "StockName": "Bank Central Asia Tbk",
            "Close": 9400,
            "Change": 50,
            "Volume": 12345678,
        }
    ],
}

_EMPTY_DATA = {"recordsTotal": 0, "recordsFiltered": 0, "data": []}

_COMPANY_DATA = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [{"KodeEmiten": "BBCA", "NamaEmiten": "Bank Central Asia Tbk"}],
}

_DAILY_DATA = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [{"IDXCode": "BBCA", "Date": "2024-01-02", "ClosePrice": 9250}],
}

_INDEX_DATA = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [{"IndexCode": "COMPOSITE", "Date": "2024-01-31", "Close": 7293.49}],
}

_BROKER_DATA = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [{"KodeBroker": "BK", "NamaBroker": "J.P. Morgan Securities Indonesia"}],
}


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------


class TestRoot:
    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# /stocks
# ---------------------------------------------------------------------------


class TestListStocks:
    def test_returns_200_with_data(self, client):
        with patch.object(app_module._client, "get_stock_summary", return_value=_STOCK_DATA):
            response = client.get("/stocks")
        assert response.status_code == 200
        assert response.json()["recordsTotal"] == 1

    def test_code_filter_forwarded(self, client):
        with patch.object(
            app_module._client, "get_stock_summary", return_value=_STOCK_DATA
        ) as mock_fn:
            client.get("/stocks?code=BBCA")
        mock_fn.assert_called_once()
        assert mock_fn.call_args.kwargs["code"] == "BBCA"

    def test_returns_502_on_error(self, client):
        with patch.object(
            app_module._client,
            "get_stock_summary",
            side_effect=Exception("upstream error"),
        ):
            response = client.get("/stocks")
        assert response.status_code == 502


class TestGetStock:
    def test_returns_stock_record(self, client):
        with patch.object(
            app_module._client,
            "get_stock_price",
            return_value=_STOCK_DATA["data"][0],
        ):
            response = client.get("/stocks/BBCA")
        assert response.status_code == 200
        assert response.json()["StockCode"] == "BBCA"

    def test_returns_404_when_not_found(self, client):
        with patch.object(app_module._client, "get_stock_price", return_value=None):
            response = client.get("/stocks/ZZZZ")
        assert response.status_code == 404

    def test_returns_502_on_error(self, client):
        with patch.object(
            app_module._client,
            "get_stock_price",
            side_effect=Exception("network error"),
        ):
            response = client.get("/stocks/BBCA")
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# /companies
# ---------------------------------------------------------------------------


class TestListCompanies:
    def test_returns_200(self, client):
        with patch.object(
            app_module._client, "get_company_profiles", return_value=_COMPANY_DATA
        ):
            response = client.get("/companies?code=BBCA")
        assert response.status_code == 200
        assert response.json()["data"][0]["KodeEmiten"] == "BBCA"

    def test_returns_502_on_error(self, client):
        with patch.object(
            app_module._client,
            "get_company_profiles",
            side_effect=Exception("err"),
        ):
            response = client.get("/companies")
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# /stocks/{code}/daily
# ---------------------------------------------------------------------------


class TestGetStockDaily:
    def test_returns_200(self, client):
        with patch.object(
            app_module._client, "get_stocks_daily", return_value=_DAILY_DATA
        ):
            response = client.get("/stocks/BBCA/daily")
        assert response.status_code == 200

    def test_date_params_forwarded(self, client):
        with patch.object(
            app_module._client, "get_stocks_daily", return_value=_DAILY_DATA
        ) as mock_fn:
            client.get("/stocks/BBCA/daily?dateFrom=2024-01-01&dateTo=2024-01-31")
        mock_fn.assert_called_once()
        assert mock_fn.call_args.kwargs["date_from"] == "2024-01-01"
        assert mock_fn.call_args.kwargs["date_to"] == "2024-01-31"

    def test_returns_502_on_error(self, client):
        with patch.object(
            app_module._client,
            "get_stocks_daily",
            side_effect=Exception("err"),
        ):
            response = client.get("/stocks/BBCA/daily")
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# /index/{index_id}
# ---------------------------------------------------------------------------


class TestGetIndexStatistics:
    def test_returns_200(self, client):
        with patch.object(
            app_module._client,
            "get_index_statistics",
            return_value=_INDEX_DATA,
        ):
            response = client.get("/index/COMPOSITE")
        assert response.status_code == 200

    def test_period_forwarded(self, client):
        with patch.object(
            app_module._client,
            "get_index_statistics",
            return_value=_INDEX_DATA,
        ) as mock_fn:
            client.get("/index/LQ45?period=monthly")
        assert mock_fn.call_args.kwargs["period"] == "monthly"


# ---------------------------------------------------------------------------
# /brokers
# ---------------------------------------------------------------------------


class TestListBrokers:
    def test_returns_200(self, client):
        with patch.object(
            app_module._client, "get_broker_summary", return_value=_BROKER_DATA
        ):
            response = client.get("/brokers")
        assert response.status_code == 200

    def test_returns_502_on_error(self, client):
        with patch.object(
            app_module._client,
            "get_broker_summary",
            side_effect=Exception("err"),
        ):
            response = client.get("/brokers")
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# /forecast/{code}
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd


def _make_price_response(n: int = 100, start: float = 9000.0) -> dict:
    prices = (start + np.arange(n) * 5).tolist()
    volumes = [1_000_000] * n
    data = [
        {
            "Date": f"2024-{(i // 30 + 1):02d}-{(i % 28 + 1):02d}",
            "ClosePrice": p,
            "Volume": v,
        }
        for i, (p, v) in enumerate(zip(prices, volumes))
    ]
    return {"recordsTotal": n, "recordsFiltered": n, "data": data}


class TestForecastStock:
    def test_returns_200_with_valid_data(self, client):
        with patch.object(
            app_module._client,
            "get_stocks_daily",
            return_value=_make_price_response(100),
        ):
            response = client.post("/forecast/BBCA")
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "BBCA"
        assert "recommendation_short_term" in body
        assert "recommendation_long_term" in body
        assert "score_short_term" in body
        assert "indicators" in body

    def test_accepts_news_headlines(self, client):
        with patch.object(
            app_module._client,
            "get_stocks_daily",
            return_value=_make_price_response(100),
        ):
            response = client.post(
                "/forecast/BBCA",
                json={"news_headlines": ["Strong profit growth", "Dividend increased"]},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment_score"] != 0.0

    def test_returns_422_when_not_enough_data(self, client):
        with patch.object(
            app_module._client,
            "get_stocks_daily",
            return_value=_make_price_response(5),
        ):
            response = client.post("/forecast/BBCA")
        assert response.status_code == 422

    def test_returns_422_when_history_days_too_small(self, client):
        response = client.post("/forecast/BBCA", json={"history_days": 30})
        assert response.status_code == 422

    def test_returns_502_on_fetch_error(self, client):
        with patch.object(
            app_module._client,
            "get_stocks_daily",
            side_effect=Exception("network error"),
        ):
            response = client.post("/forecast/BBCA")
        assert response.status_code == 502

    def test_code_is_uppercased(self, client):
        with patch.object(
            app_module._client,
            "get_stocks_daily",
            return_value=_make_price_response(100),
        ):
            response = client.post("/forecast/bbca")
        assert response.status_code == 200
        assert response.json()["code"] == "BBCA"
