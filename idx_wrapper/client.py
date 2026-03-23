"""IDX API client for fetching Indonesia Stock Exchange data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

BASE_URL = "https://www.idx.co.id"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IDX-py-wrapper/1.0; "
        "+https://github.com/salmanhiro/IDX-py-wrapper)"
    ),
    "Accept": "application/json",
}


class IDXClient:
    """Client for the public Indonesia Stock Exchange (IDX) APIs.

    Parameters
    ----------
    timeout:
        Request timeout in seconds (default 30).
    session:
        Optional pre-configured :class:`requests.Session` to use.  Useful
        for injecting test fixtures.

    Example
    -------
    >>> from idx_wrapper import IDXClient
    >>> client = IDXClient()
    >>> summary = client.get_stock_summary(code="BBCA")
    """

    def __init__(
        self,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform a GET request and return parsed JSON."""
        url = f"{BASE_URL}{path}"
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Stock summary
    # ------------------------------------------------------------------

    def get_stock_summary(
        self,
        code: str = "",
        name: str = "",
        start: int = 0,
        length: int = 100,
    ) -> Dict[str, Any]:
        """Return trading summary data for listed stocks.

        Parameters
        ----------
        code:
            Stock ticker code filter (e.g. ``"BBCA"``).  Leave empty for all.
        name:
            Company name filter.  Leave empty for all.
        start:
            Pagination offset (0-based).
        length:
            Number of records to return (max 9999).

        Returns
        -------
        dict
            JSON response with ``recordsTotal``, ``recordsFiltered``, and
            ``data`` keys.
        """
        params = {
            "start": start,
            "length": length,
            "code": code,
            "name": name,
        }
        return self._get("/primary/TradingSummary/GetStockSummary", params=params)

    # ------------------------------------------------------------------
    # Listed companies
    # ------------------------------------------------------------------

    def get_company_profiles(
        self,
        code: str = "",
        name: str = "",
        start: int = 0,
        length: int = 100,
    ) -> Dict[str, Any]:
        """Return profiles for companies listed on IDX.

        Parameters
        ----------
        code:
            Stock ticker code filter.
        name:
            Company name filter.
        start:
            Pagination offset.
        length:
            Number of records to return.

        Returns
        -------
        dict
            JSON response containing company profile data.
        """
        params = {
            "start": start,
            "length": length,
            "code": code,
            "name": name,
        }
        return self._get(
            "/primary/ListedCompany/GetCompanyProfiles", params=params
        )

    # ------------------------------------------------------------------
    # Daily stock data
    # ------------------------------------------------------------------

    def get_stocks_daily(
        self,
        code: str = "",
        date_from: str = "",
        date_to: str = "",
        start: int = 0,
        length: int = 100,
    ) -> Dict[str, Any]:
        """Return daily OHLCV data for one or all stocks.

        Parameters
        ----------
        code:
            Stock ticker code filter.
        date_from:
            Start date in ``YYYY-MM-DD`` format.
        date_to:
            End date in ``YYYY-MM-DD`` format.
        start:
            Pagination offset.
        length:
            Number of records to return.

        Returns
        -------
        dict
            JSON response with daily price data.
        """
        params = {
            "start": start,
            "length": length,
            "code": code,
            "dateFrom": date_from,
            "dateTo": date_to,
        }
        return self._get("/primary/StockData/GetStocksDaily", params=params)

    # ------------------------------------------------------------------
    # Index statistics
    # ------------------------------------------------------------------

    def get_index_statistics(
        self,
        index_id: str = "COMPOSITE",
        period: str = "daily",
        start: int = 0,
        length: int = 100,
    ) -> Dict[str, Any]:
        """Return historical statistics for a market index.

        Parameters
        ----------
        index_id:
            Index identifier (e.g. ``"COMPOSITE"`` for IHSG, ``"LQ45"``).
        period:
            Data frequency: ``"daily"``, ``"monthly"``, or ``"yearly"``.
        start:
            Pagination offset.
        length:
            Number of records to return.

        Returns
        -------
        dict
            JSON response with index time series data.
        """
        params = {
            "indexId": index_id,
            "period": period,
            "start": start,
            "length": length,
        }
        return self._get(
            "/primary/IndexStatistic/GetIndexStatisticData", params=params
        )

    # ------------------------------------------------------------------
    # Broker summary
    # ------------------------------------------------------------------

    def get_broker_summary(
        self,
        start: int = 0,
        length: int = 100,
    ) -> Dict[str, Any]:
        """Return trading summary per broker member.

        Parameters
        ----------
        start:
            Pagination offset.
        length:
            Number of records to return.

        Returns
        -------
        dict
            JSON response with broker trading data.
        """
        params = {"start": start, "length": length}
        return self._get(
            "/primary/TradingSummary/GetBrokerSummary", params=params
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_all_stocks(self) -> List[Dict[str, Any]]:
        """Fetch the complete list of stock summaries (all records).

        Returns
        -------
        list
            All records from the stock summary endpoint.
        """
        result = self.get_stock_summary(length=9999)
        return result.get("data", [])

    def get_stock_price(self, code: str) -> Optional[Dict[str, Any]]:
        """Return the latest trading summary for a single stock code.

        Parameters
        ----------
        code:
            Stock ticker code (e.g. ``"BBCA"``).

        Returns
        -------
        dict or None
            The first matching record, or ``None`` if not found.
        """
        result = self.get_stock_summary(code=code.upper(), length=1)
        data: List[Dict[str, Any]] = result.get("data", [])
        return data[0] if data else None
