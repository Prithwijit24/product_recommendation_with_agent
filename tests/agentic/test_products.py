import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import products


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload

    # mock: httpx semantic for raise_for_status missing in brief scaffold — controller-approved
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_serp_google_shopping_parse(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(
            {
                "shopping_results": [
                    {
                        "title": "CeraVe PM Facial Moisturizing Lotion",
                        "price": "$14.99",
                        "link": "https://cerave.com/pm",
                        "source": "cerave.com",
                    }
                ]
            }
        )

    monkeypatch.setattr(products.httpx, "get", fake_get)
    out = products._serp_search("niacinamide moisturizer", "KEY")
    assert captured["params"]["engine"] == "google_shopping"
    assert out[0]["name"] == "CeraVe PM Facial Moisturizing Lotion"
    assert out[0]["price"] == "$14.99"
    assert out[0]["url"] == "https://cerave.com/pm"


def test_find_products_budget_low_filters(monkeypatch):
    monkeypatch.setenv("SERP_API_KEY", "KEY")

    def fake_aistack_search(ingredient, client=None):
        return [
            {"name": "Luxury A Serum", "url": "http://a", "price": "$120.00", "source": "aistack"},
            {"name": "Budget B Serum", "url": "http://b", "price": "$9.99", "source": "aistack"},
        ]

    monkeypatch.setattr(products, "_aistack_search", fake_aistack_search)
    out = products.find_products(["hyaluronic acid"], budget="low")
    prices = [float(p["price"].replace("$", "")) for p in out]
    assert len(out) >= 1
    assert max(prices) <= 30.0


def test_find_products_keeps_source_and_match(monkeypatch):
    monkeypatch.setenv("SERP_API_KEY", "KEY")

    def fake_aistack_search(ingredient, client=None):
        return [
            {"name": "The Ordinary Niacinamide 10%", "url": "http://to", "price": "$6.99", "source": "aistack"},
        ]

    monkeypatch.setattr(products, "_aistack_search", fake_aistack_search)
    out = products.find_products(["niacinamide"], budget="medium")
    assert out[0]["source"]
    assert "niacinamide" in out[0]["ingredient_match"].lower()


def test_serp_error_degrades_to_fallback(monkeypatch):
    monkeypatch.setenv("SERP_API_KEY", "KEY")

    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(products.httpx, "get", fake_get)

    def fake_obf(query):
        return [{"name": "Fragrance Fixer", "url": "http://obf", "price": "", "source": "openbeautyfacts"}]

    monkeypatch.setattr(products, "_open_beauty_facts", fake_obf)
    out = products.find_products(["fragrance"], budget="medium")
    assert len(out) == 1
    assert out[0]["name"] == "Fragrance Fixer"
    assert out[0]["source"] == "openbeautyfacts"


def test_all_sources_fail_returns_empty(monkeypatch):
    """When all product sources fail, find_products should return empty list."""

    def fake_get(url=None, params=None, timeout=None, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    # Empty AISTACK env vars so aistack search fails
    monkeypatch.setenv("AISTACK_BASE_URL", "")
    monkeypatch.setenv("AISTACK_API_KEY", "")
    monkeypatch.setenv("BASE_URL", "")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("SERP_API_KEY", "KEY")
    out = products.find_products(["fragrance"], budget="medium")
    assert out == []