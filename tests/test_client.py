"""Tests for IDXClient using mocked HTTP responses."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from idx_wrapper import IDXClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response with JSON body."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


_STOCK_SUMMARY_RESPONSE = {
    "draw": 1,
    "recordsTotal": 2,
    "recordsFiltered": 2,
    "data": [
        {
            "StockCode": "BBCA",
            "StockName": "Bank Central Asia Tbk",
            "Remarks": "",
            "Previous": 9350,
            "OpenPrice": 9400,
            "FirstTrade": 9400,
            "High": 9450,
            "Low": 9350,
            "Close": 9400,
            "Change": 50,
            "Volume": 12345678,
            "Value": 115987654321,
            "Frequency": 15000,
            "IndexIndividual": 0,
            "Offer": 9425,
            "OfferVolume": 100,
            "Bid": 9400,
            "BidVolume": 200,
            "ListedShares": 1234567890,
            "TradableShares": 987654321,
            "WeightForIndex": 1.234,
            "ForeignSell": 500000,
            "ForeignBuy": 600000,
            "DelistingDate": None,
            "NonRegularVolume": 0,
            "NonRegularValue": 0,
            "NonRegularFrequency": 0,
        },
        {
            "StockCode": "TLKM",
            "StockName": "Telkom Indonesia (Persero) Tbk",
            "Remarks": "",
            "Previous": 3890,
            "OpenPrice": 3900,
            "FirstTrade": 3900,
            "High": 3950,
            "Low": 3880,
            "Close": 3920,
            "Change": 30,
            "Volume": 9876543,
            "Value": 38668988560,
            "Frequency": 10000,
            "IndexIndividual": 0,
            "Offer": 3930,
            "OfferVolume": 50,
            "Bid": 3920,
            "BidVolume": 150,
            "ListedShares": 999888777,
            "TradableShares": 888777666,
            "WeightForIndex": 0.987,
            "ForeignSell": 300000,
            "ForeignBuy": 350000,
            "DelistingDate": None,
            "NonRegularVolume": 0,
            "NonRegularValue": 0,
            "NonRegularFrequency": 0,
        },
    ],
}

_COMPANY_PROFILES_RESPONSE = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [
        {
            "KodeEmiten": "BBCA",
            "NamaEmiten": "Bank Central Asia Tbk",
            "Sektor": "Finance",
            "Subsector": "Banks",
            "ListingDate": "2000-05-31",
            "Shares": 1234567890,
            "MarketCap": 11590123456000,
        }
    ],
}

_DAILY_RESPONSE = {
    "recordsTotal": 2,
    "recordsFiltered": 2,
    "data": [
        {
            "IDXCode": "BBCA",
            "Date": "2024-01-02",
            "OpenPrice": 9200,
            "HighPrice": 9300,
            "LowPrice": 9150,
            "ClosePrice": 9250,
            "Volume": 10000000,
            "Value": 92500000000,
            "Frequency": 12000,
            "Change": 50,
            "PercentChange": 0.54,
        },
        {
            "IDXCode": "BBCA",
            "Date": "2024-01-03",
            "OpenPrice": 9250,
            "HighPrice": 9350,
            "LowPrice": 9200,
            "ClosePrice": 9300,
            "Volume": 11000000,
            "Value": 102300000000,
            "Frequency": 13000,
            "Change": 50,
            "PercentChange": 0.54,
        },
    ],
}

_INDEX_RESPONSE = {
    "recordsTotal": 2,
    "recordsFiltered": 2,
    "data": [
        {"IndexCode": "COMPOSITE", "Date": "2024-01-31", "Close": 7293.49},
        {"IndexCode": "COMPOSITE", "Date": "2024-02-29", "Close": 7351.14},
    ],
}

_BROKER_RESPONSE = {
    "recordsTotal": 1,
    "recordsFiltered": 1,
    "data": [
        {
            "KodeBroker": "BK",
            "NamaBroker": "J.P. Morgan Securities Indonesia",
            "BuyValue": 1000000000,
            "SellValue": 950000000,
        }
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIDXClientInit:
    def test_default_initialization(self):
        client = IDXClient()
        assert client.timeout == 30
        assert client._session is not None

    def test_custom_timeout(self):
        client = IDXClient(timeout=60)
        assert client.timeout == 60

    def test_custom_session(self):
        session = requests.Session()
        client = IDXClient(session=session)
        assert client._session is session


class TestGetStockSummary:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_data(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        result = self.client.get_stock_summary()
        assert result["recordsTotal"] == 2
        assert len(result["data"]) == 2

    def test_passes_code_filter(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        self.client.get_stock_summary(code="BBCA")
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["code"] == "BBCA"

    def test_passes_name_filter(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        self.client.get_stock_summary(name="bank")
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["name"] == "bank"

    def test_passes_pagination(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        self.client.get_stock_summary(start=10, length=50)
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["start"] == 10
        assert kwargs["params"]["length"] == 50

    def test_raises_on_http_error(self):
        error_response = MagicMock(spec=requests.Response)
        error_response.raise_for_status.side_effect = requests.HTTPError("503")
        self.mock_session.get.return_value = error_response
        with pytest.raises(requests.HTTPError):
            self.client.get_stock_summary()


class TestGetStockPrice:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_first_record(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        result = self.client.get_stock_price("BBCA")
        assert result is not None
        assert result["StockCode"] == "BBCA"

    def test_returns_none_when_no_data(self):
        empty = {"recordsTotal": 0, "recordsFiltered": 0, "data": []}
        self.mock_session.get.return_value = _make_response(empty)
        result = self.client.get_stock_price("ZZZZ")
        assert result is None

    def test_uppercases_code(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        self.client.get_stock_price("bbca")
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["code"] == "BBCA"


class TestGetAllStocks:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_list(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        result = self.client.get_all_stocks()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_requests_max_length(self):
        self.mock_session.get.return_value = _make_response(_STOCK_SUMMARY_RESPONSE)
        self.client.get_all_stocks()
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["length"] == 9999


class TestGetCompanyProfiles:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_data(self):
        self.mock_session.get.return_value = _make_response(
            _COMPANY_PROFILES_RESPONSE
        )
        result = self.client.get_company_profiles(code="BBCA")
        assert result["recordsTotal"] == 1
        assert result["data"][0]["KodeEmiten"] == "BBCA"

    def test_correct_endpoint(self):
        self.mock_session.get.return_value = _make_response(
            _COMPANY_PROFILES_RESPONSE
        )
        self.client.get_company_profiles()
        args, _ = self.mock_session.get.call_args
        assert "/primary/ListedCompany/GetCompanyProfiles" in args[0]


class TestGetStocksDaily:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_data(self):
        self.mock_session.get.return_value = _make_response(_DAILY_RESPONSE)
        result = self.client.get_stocks_daily(
            code="BBCA", date_from="2024-01-01", date_to="2024-01-31"
        )
        assert result["recordsTotal"] == 2
        assert result["data"][0]["IDXCode"] == "BBCA"

    def test_passes_date_params(self):
        self.mock_session.get.return_value = _make_response(_DAILY_RESPONSE)
        self.client.get_stocks_daily(
            code="BBCA", date_from="2024-01-01", date_to="2024-01-31"
        )
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["dateFrom"] == "2024-01-01"
        assert kwargs["params"]["dateTo"] == "2024-01-31"


class TestGetIndexStatistics:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_data(self):
        self.mock_session.get.return_value = _make_response(_INDEX_RESPONSE)
        result = self.client.get_index_statistics(
            index_id="COMPOSITE", period="monthly"
        )
        assert result["recordsTotal"] == 2

    def test_passes_index_params(self):
        self.mock_session.get.return_value = _make_response(_INDEX_RESPONSE)
        self.client.get_index_statistics(index_id="LQ45", period="daily")
        _, kwargs = self.mock_session.get.call_args
        assert kwargs["params"]["indexId"] == "LQ45"
        assert kwargs["params"]["period"] == "daily"


class TestGetBrokerSummary:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.client = IDXClient(session=self.mock_session)

    def test_returns_data(self):
        self.mock_session.get.return_value = _make_response(_BROKER_RESPONSE)
        result = self.client.get_broker_summary()
        assert result["recordsTotal"] == 1

    def test_correct_endpoint(self):
        self.mock_session.get.return_value = _make_response(_BROKER_RESPONSE)
        self.client.get_broker_summary()
        args, _ = self.mock_session.get.call_args
        assert "/primary/TradingSummary/GetBrokerSummary" in args[0]
