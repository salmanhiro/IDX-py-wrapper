"""News headline fetching helpers for the IDX forecasting tools."""

from __future__ import annotations

from typing import List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

from .version import __version__

_DEFAULT_LANG = "en"
_DEFAULT_REGION = "ID"
MAX_NEWS_LOOKBACK_DAYS = 30


def _build_google_news_url(query: str, days: int, lang: str, region: str) -> str:
    search = f"{query} when:{days}d"
    encoded = quote_plus(search)
    return (
        "https://news.google.com/rss/search?"
        f"q={encoded}&hl={lang}&gl={region}&ceid={region}:{lang}"
    )


def _parse_rss_titles(xml_text: str, limit: int) -> List[str]:
    root = ET.fromstring(xml_text)
    titles: List[str] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title")
        if title:
            titles.append(title.strip())
        if len(titles) >= limit:
            break
    return titles


def fetch_latest_headlines(
    query: str,
    limit: int = 10,
    days: int = 1,
    *,
    lang: str = _DEFAULT_LANG,
    region: str = _DEFAULT_REGION,
    timeout: float = 8.0,
) -> List[str]:
    """Fetch latest headlines for a query using Google News RSS.

    Parameters
    ----------
    query:
        Search query (e.g. "BBCA stock").
    limit:
        Maximum number of headlines to return.
    days:
        Lookback window in days (Google News "when:Xd" query). Values above
        MAX_NEWS_LOOKBACK_DAYS are clamped.
    lang / region:
        Language/region settings for the RSS feed.
    timeout:
        Requests timeout in seconds.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if days < 1:
        raise ValueError("days must be at least 1.")

    clamped_days = min(days, MAX_NEWS_LOOKBACK_DAYS)
    url = _build_google_news_url(query.strip(), clamped_days, lang, region)
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": f"IDX-py-wrapper/{__version__} (+https://github.com/salmanhiro/IDX-py-wrapper)"
        },
    )
    response.raise_for_status()
    return _parse_rss_titles(response.text, limit)
