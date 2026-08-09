"""Thin httpx client for the aistack API (search/crawl/health)."""

from __future__ import annotations

import os

import httpx

from . import config


def _cfg() -> dict:
    return config.get_section("aistack")


class AistackClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        cfg = _cfg()
        # Support both new (AISTACK_*) and legacy (BASE_URL/API_KEY) env var names
        self.base_url = (
            base_url
            or os.getenv(cfg.get("base_url_env", "AISTACK_BASE_URL"), "")
            or os.getenv("BASE_URL", "")
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv(cfg.get("api_key_env", "AISTACK_API_KEY"), "")
            or os.getenv("API_KEY", "")
        )
        self._http = httpx.Client(timeout=float(cfg.get("http_timeout", 90.0)))

    def _headers(self) -> dict:
        cfg = _cfg()
        return {"Content-Type": "application/json", cfg.get("auth_header", "X-API-Key"): self.api_key}

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        cfg = _cfg()
        if not self.base_url:
            raise RuntimeError("AISTACK_BASE_URL not configured")
        if max_results is None:
            max_results = int(cfg.get("default_max_results", 5))
        endpoints = cfg.get("endpoints", {})
        resp = self._http.post(
            f"{self.base_url}{endpoints.get('search', '/search')}",
            headers=self._headers(),
            json={"query": query, "max_results": max_results},
        )
        if resp.status_code != 200:
            truncate = int(cfg.get("error_text_truncate", 200))
            raise RuntimeError(f"aistack /search: HTTP {resp.status_code} {resp.text[:truncate]}")
        return resp.json().get("results", [])

    def crawl(self, url: str, only_main_content: bool = True) -> str:
        cfg = _cfg()
        if not self.base_url:
            raise RuntimeError("AISTACK_BASE_URL not configured")
        endpoints = cfg.get("endpoints", {})
        resp = self._http.post(
            f"{self.base_url}{endpoints.get('crawl', '/crawl')}",
            headers=self._headers(),
            json={"url": url, "only_main_content": only_main_content},
        )
        if resp.status_code != 200:
            truncate = int(cfg.get("error_text_truncate", 200))
            raise RuntimeError(f"aistack /crawl: HTTP {resp.status_code} {resp.text[:truncate]}")
        return resp.json().get("markdown", "")

    def health(self) -> dict:
        if not self.base_url:
            raise RuntimeError("AISTACK_BASE_URL not configured")
        cfg = _cfg()
        endpoints = cfg.get("endpoints", {})
        resp = self._http.get(f"{self.base_url}{endpoints.get('health', '/health')}", headers=self._headers())
        if resp.status_code != 200:
            raise RuntimeError(f"aistack /health: HTTP {resp.status_code}")
        return resp.json()
