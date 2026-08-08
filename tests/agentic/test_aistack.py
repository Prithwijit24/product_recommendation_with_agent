import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import aistack


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload


def test_search_posts_to_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse({"number_of_results": 1, "results": [{"url": "http://x", "title": "t"}]})

    monkeypatch.setenv("BASE_URL", "https://aistack.example")
    monkeypatch.setenv("API_KEY", "k")
    client = aistack.AistackClient()
    monkeypatch.setattr(client._http, "post", fake_post)
    out = client.search("niacinamide", max_results=3)
    assert captured["url"].endswith("/search")
    assert captured["json"] == {"query": "niacinamide", "max_results": 3}
    assert captured["headers"]["X-API-Key"] == "k"
    assert out[0]["title"] == "t"


def test_health_live(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://aistack.example")
    monkeypatch.setenv("API_KEY", "k")
    client = aistack.AistackClient()

    def fake_get(url, headers, timeout=None):
        return FakeResponse({"status": "ok"})

    monkeypatch.setattr(client._http, "get", fake_get)
    assert client.health()["status"] == "ok"
