import sys
from pathlib import Path

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

    def fake_get(url, params, timeout=None):
        return FakeResponse(
            {
                "shopping_results": [
                    {"title": "Luxury A", "price": "$120.00", "link": "http://a", "source": "x"},
                    {"title": "Budget B", "price": "$9.99", "link": "http://b", "source": "y"},
                ]
            }
        )

    monkeypatch.setattr(products.httpx, "get", fake_get)
    out = products.find_products(["hyaluronic acid"], budget="low")
    prices = [float(p["price"].replace("$", "")) for p in out]
    assert len(out) >= 1
    assert max(prices) <= 30.0


def test_find_products_keeps_source_and_match(monkeypatch):
    monkeypatch.setenv("SERP_API_KEY", "KEY")

    def fake_get(url, params, timeout=None):
        return FakeResponse(
            {
                "shopping_results": [
                    {"title": "The Ordinary Niacinamide 10%", "price": "$6.99", "link": "http://to", "source": "ordinary.com"},
                ]
            }
        )

    monkeypatch.setattr(products.httpx, "get", fake_get)
    out = products.find_products(["niacinamide"], budget="medium")
    assert out[0]["source"]
    assert "niacinamide" in out[0]["ingredient_match"].lower()