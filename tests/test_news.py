"""Tests for RSS-based news headline fetching."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from idx_wrapper.news import fetch_latest_headlines

_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sample</title>
    <item><title>Headline A</title></item>
    <item><title>Headline B</title></item>
    <item><title>Headline C</title></item>
  </channel>
</rss>
"""


def test_fetch_latest_headlines_parses_titles():
    response = Mock()
    response.text = _SAMPLE_RSS
    response.raise_for_status = Mock()

    with patch("idx_wrapper.news.requests.get", return_value=response) as mock_get:
        headlines = fetch_latest_headlines("BBCA stock", limit=2, days=3)

    assert headlines == ["Headline A", "Headline B"]
    called_url = mock_get.call_args[0][0]
    assert "when%3A3d" in called_url


def test_fetch_latest_headlines_requires_query():
    with pytest.raises(ValueError):
        fetch_latest_headlines("")


def test_fetch_latest_headlines_requires_positive_limit():
    with pytest.raises(ValueError):
        fetch_latest_headlines("BBCA", limit=0)
