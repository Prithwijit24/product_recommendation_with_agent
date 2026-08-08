"""Thin httpx client for the aistack API (search/crawl/health)."""

from __future__ import annotations

import os

import httpx


class AistackClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.getenv("BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY", "")
        self._http = httpx.Client(timeout=90.0)

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "X-API-Key": self.api_key}

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        if not self.base_url:
            raise RuntimeError("BASE_URL not configured")
        resp = self._http.post(
            f"{self.base_url}/search",
            headers=self._headers(),
            json={"query": query, "max_results": max_results},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"aistack /search: HTTP {resp.status_code}")
        return resp.json().get("results", [])

    def crawl(self, url: str, only_main_content: bool = True) -> str:
        if not self.base_url:
            raise RuntimeError("BASE_URL not configured")
        resp = self._http.post(
            f"{self.base_url}/crawl",
            headers=self._headers(),
            json={"url": url, "only_main_content": only_main_content},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"aistack /crawl: HTTP {resp.status_code}")
        return resp.json().get("markdown", "")

    def health(self) -> dict:
        if not self.base_url:
            raise RuntimeError("BASE_URL not configured")
        resp = self._http.get(f"{self.base_url}/health", headers=self._headers())
        if resp.status_code != 200:
            raise RuntimeError(f"aistack /health: HTTP {resp.status_code}")
        return resp.json()