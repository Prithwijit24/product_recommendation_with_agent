"""Product discovery: SerpAPI google_shopping first; aistack /search; Open Beauty Facts last."""

from __future__ import annotations

import os

import httpx

from . import aistack

SERPAPI_URL = "https://serpapi.com/search.json"
BUDGET_LIMIT = {"low": 30.0, "medium": 90.0, "high": float("inf")}


def _parse_price(raw: str) -> float | None:
    if not raw:
        return None
    cleaned = "".join(c for c in raw if c.isdigit() or c in ".,")
    try:
        return float(cleaned.replace(",", "."))
    except ValueError:
        return None


def _serp_search(query: str, serp_api_key: str | None = None) -> list[dict]:
    key = serp_api_key or os.getenv("SERP_API_KEY", "")
    if not key:
        return []
    resp = httpx.get(
        SERPAPI_URL,
        params={"engine": "google_shopping", "q": query, "api_key": key},
        timeout=20.0,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json().get("shopping_results", []):
        out.append(
            {
                "name": item.get("title", ""),
                "url": item.get("link", ""),
                "price": item.get("price", ""),
                "source": item.get("source", ""),
            }
        )
    return out


def _open_beauty_facts(query: str) -> list[dict]:
    try:
        resp = httpx.get(
            "https://world.openbeautyfacts.org/cgi/search.pl",
            params={
                "action": "process",
                "search_terms": query,
                "json": 1,
                "fields": "product_name,brands,_id",
            },
            timeout=15.0,
        )
        data = resp.json()
    except Exception:
        return []
    out = []
    for p in data.get("products", [])[:5]:
        pid = p.get("_id", "")
        out.append(
            {
                "name": p.get("product_name", "") or "",
                "url": f"https://world.openbeautyfacts.org/product/{pid}" if pid else "",
                "price": "",
                "source": "openbeautyfacts",
            }
        )
    return out


def _aistack_fallback(ingredient: str, client: aistack.AistackClient | None = None) -> list[dict]:
    try:
        c = client or aistack.AistackClient()
        results = c.search(f"{ingredient} buy skincare", max_results=3)
    except Exception:
        return []
    return [
        {
            "name": r.get("title", ""),
            "url": r.get("url", ""),
            "price": "",
            "source": "aistack-search",
        }
        for r in results
        if r.get("url")
    ]


def _match(ing: str, title: str) -> str:
    tl = title.lower()
    for token in ing.lower().replace("-", " ").split():
        if token in tl:
            return f"{ing} (found in title)"
    return f"{ing} (best available match)"


def find_products(ingredients: list[str], budget: str = "medium") -> list[dict]:
    """One product per ingredient — best cascade source wins. Never invents anything."""
    limit = BUDGET_LIMIT.get(budget, BUDGET_LIMIT["medium"])
    result: list[dict] = []
    for ing in ingredients:
        found: list[dict] = []
        try:
            for q in (f"{ing} skincare", ing):
                found = _serp_search(q)
                if found:
                    break
        except Exception:
            found = []
        if not found:
            found = _aistack_fallback(ing)
        if not found:
            found = _open_beauty_facts(ing)
        if not found:
            continue
        for item in found:
            price_f = _parse_price(item.get("price", ""))
            if price_f is not None and price_f > limit:
                continue
            result.append(
                {
                    "name": item.get("name", ""),
                    "url": item.get("url", ""),
                    "price": item.get("price", "") or "",
                    "ingredient": ing,
                    "ingredient_match": _match(ing, item.get("name", "")),
                    "source": item.get("source", ""),
                }
            )
            break  # first acceptable product per ingredient
    return result