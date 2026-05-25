"""Tests for WeiboAPIScraper using mocked httpx."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from beverage_ai.scrapers.adapters.weibo_api import (
    WeiboAPIError,
    WeiboAPIScraper,
    _status_to_record,
)


def test_constructor_requires_token(monkeypatch):
    monkeypatch.delenv("WEIBO_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="WEIBO_ACCESS_TOKEN"):
        WeiboAPIScraper()


def test_constructor_accepts_token_arg():
    s = WeiboAPIScraper(access_token="dummy_token_xyz")
    assert s.token == "dummy_token_xyz"


def test_status_to_record_basic():
    rec = _status_to_record(
        {"id": 12345, "text": "今天买的奶茶味道超级好喝,推荐", "user": {"id": "abc"}},
        brand="喜茶",
    )
    assert rec is not None
    assert rec.brand == "喜茶"
    assert rec.source == "weibo_api"
    assert "12345" in rec.source_url
    assert rec.user_id_hashed is not None
    assert not rec.user_id_hashed.startswith("abc")    # hashed, not raw


def test_status_to_record_skips_short_text():
    rec = _status_to_record({"text": "好", "user": {"id": "x"}}, brand=None)
    assert rec is None


def test_status_to_record_skips_empty_text():
    rec = _status_to_record({"text": "", "user": {}}, brand=None)
    assert rec is None


def test_scrape_uses_httpx_and_returns_records(monkeypatch):
    """Mock httpx; verify pagination + record conversion."""
    fake_resp_page1 = MagicMock()
    fake_resp_page1.status_code = 200
    fake_resp_page1.json.return_value = {
        "statuses": [
            {"id": 1, "text": "今天买的奶茶味道超级好喝,推荐", "user": {"id": "u1"}},
            {"id": 2, "text": "桂花乌龙拿铁非常香醇值得回购", "user": {"id": "u2"}},
        ],
    }
    fake_resp_page2 = MagicMock()
    fake_resp_page2.status_code = 200
    fake_resp_page2.json.return_value = {"statuses": []}    # signal end

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None
    fake_client.get.side_effect = [fake_resp_page1, fake_resp_page2]

    fake_httpx = MagicMock()
    fake_httpx.Client.return_value = fake_client
    fake_httpx.HTTPError = Exception
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    s = WeiboAPIScraper(access_token="dummy", max_per_hour=10000)
    # Pace down to avoid the test sleeping a long time
    s.max_per_hour = 100000
    records = list(s.scrape(keywords=["奶茶"], max_records=10))
    assert len(records) == 2


def test_scrape_raises_on_api_error(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"error": "invalid token", "error_code": 21314}

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None
    fake_client.get.return_value = fake_resp

    fake_httpx = MagicMock()
    fake_httpx.Client.return_value = fake_client
    fake_httpx.HTTPError = Exception
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    s = WeiboAPIScraper(access_token="dummy", max_per_hour=100000)
    with pytest.raises(WeiboAPIError, match="invalid token"):
        list(s.scrape(keywords=["奶茶"]))


def test_scrape_raises_on_http_status_error(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.status_code = 429
    fake_resp.text = "rate limited"

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None
    fake_client.get.return_value = fake_resp

    fake_httpx = MagicMock()
    fake_httpx.Client.return_value = fake_client
    fake_httpx.HTTPError = Exception
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    s = WeiboAPIScraper(access_token="dummy", max_per_hour=100000)
    with pytest.raises(WeiboAPIError, match="429"):
        list(s.scrape(keywords=["奶茶"]))


def test_raises_clear_error_without_httpx(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    s = WeiboAPIScraper(access_token="dummy")
    with pytest.raises(ImportError, match=r"\[scrape\]"):
        list(s.scrape(keywords=["奶茶"]))
