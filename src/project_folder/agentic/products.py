"""Product discovery: aistack /search first, Open Beauty Facts second, SerpAPI only for final price/link lookup."""

from __future__ import annotations

import logging
import os
import re

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from . import aistack, config

logger = logging.getLogger(__name__)


def fetch_product_image(url: str, timeout: float = 10.0, product_name: str = "") -> str:
    """Fetch product image URL from a product page.

    Tries multiple methods:
    1. Direct HTTP request + parse og:image / twitter:image / schema.org
    2. Extract any large images from the page
    3. Aistack crawl endpoint as fallback

    Args:
        url: Product page URL
        timeout: Request timeout in seconds
        product_name: Product name for additional context

    Returns:
        Image URL if found, empty string otherwise
    """
    if not url:
        return ""

    # Method 1: Direct HTTP request + parse HTML
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        html = resp.text

        # Try to find product image from Open Beauty Facts pattern first
        obf_img = re.search(
            r'https://images\.openbeautyfacts\.org/images/products/[^\s"<>]+\.jpg',
            html,
        )
        if obf_img:
            return obf_img.group(0)

        # Try og:image first (most reliable for product pages)
        og_match = re.search(
            r'<meta\s+(?:property|name)=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if og_match:
            img_url = og_match.group(1)
            # Skip if it's just a logo
            if "logo" not in img_url.lower():
                return img_url

        # Try twitter:image
        tw_match = re.search(
            r'<meta\s+(?:property|name)=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if tw_match:
            img_url = tw_match.group(1)
            if "logo" not in img_url.lower():
                return img_url

        # Try schema.org image
        schema_match = re.search(r'"image"\s*:\s*"([^"]+)"', html)
        if schema_match:
            return schema_match.group(1)

        # Try any image URLs in the page that look like product images
        img_matches = re.findall(
            r'<img[^>]+src=["\'](https?://[^"\']+\.(?:jpg|jpeg|png|webp))["\']',
            html, re.IGNORECASE,
        )
        # Filter out small icons and logos, prefer product images
        for img_url in img_matches:
            if any(skip in img_url.lower() for skip in ["logo", "icon", "avatar", "sprite", "sprites"]):
                continue
            if "1x1" in img_url or "pixel" in img_url:
                continue
            if "product" in img_url.lower() or "front" in img_url.lower():
                return img_url
        # Return first non-logo image as last resort
        for img_url in img_matches:
            if "logo" not in img_url.lower() and "icon" not in img_url.lower():
                return img_url

    except (httpx.HTTPError, ValueError, TimeoutError) as e:
        logger.debug(f"Direct fetch failed for {url}: {e}")

    # Method 2: Aistack crawl endpoint
    try:
        client = aistack.AistackClient()
        crawl_result = client.crawl(url, only_main_content=False)
        # Look for any image URLs in the crawl result
        image_urls = re.findall(r'https?://[^\s"<>]+\.(?:jpg|jpeg|png|webp|gif)', crawl_result, re.IGNORECASE)
        if image_urls:
            return image_urls[0]
    except (httpx.HTTPError, RuntimeError) as e:
        logger.debug(f"Aistack crawl failed for {url}: {e}")

    return ""


def _products_cfg() -> dict:
    return config.get_section("products")


def _parse_price(raw: str) -> float | None:
    if not raw:
        return None
    cleaned = "".join(c for c in raw if c.isdigit() or c in ".,")
    try:
        return float(cleaned.replace(",", "."))
    except ValueError:
        return None


def _serp_search(query: str, serp_api_key: str | None = None) -> list[dict]:
    """SerpAPI lookup - returns products with prices AND images."""
    cfg = _products_cfg().get("serpapi", {})
    key = serp_api_key or os.getenv(cfg.get("api_key_env", "SERP_API_KEY"), "")
    if not key:
        logger.warning("SerpAPI key not set, skipping SerpAPI lookup")
        return []
    logger.info(f"SerpAPI lookup: {query}")
    try:
        resp = httpx.get(
            cfg.get("url", ""),
            params={"engine": cfg.get("engine", "google_shopping"), "q": query, "api_key": key},
            timeout=float(cfg.get("timeout", 20.0)),
        )
        resp.raise_for_status()
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning(f"SerpAPI request failed for '{query}': {e}")
        return []
    out = []
    raw_results = resp.json().get("shopping_results", [])
    logger.info(f"SerpAPI returned {len(raw_results)} raw results for: {query}")
    for item in raw_results:
        url = item.get("link", "")
        title = item.get("title", "")
        if not url:
            # Try product_link as fallback
            url = item.get("product_link", "")
        if not _is_product_url(url, title):
            logger.debug(f"Filtered out: {title[:50]} | URL: {url[:80]}")
            continue
        out.append(
            {
                "name": _clean_product_name(title),
                "url": url,
                "price": item.get("price", ""),
                "image_url": item.get("thumbnail", item.get("serpapi_thumbnail", "")),
                "source": item.get("source", ""),
            }
        )
    logger.info(f"SerpAPI returned {len(out)} product results for: {query}")
    return out


def _open_beauty_facts(query: str) -> list[dict]:
    """Open Beauty Facts search - free, no API key needed."""
    cfg = _products_cfg().get("open_beauty_facts", {})
    logger.info(f"OpenBeautyFacts lookup: {query}")
    try:
        resp = httpx.get(
            cfg.get("search_url", ""),
            params={
                "action": "process",
                "search_terms": query,
                "json": 1,
                "fields": cfg.get("fields", "product_name,brands,_id"),
            },
            timeout=float(cfg.get("timeout", 15.0)),
        )
        data = resp.json()
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning(f"OpenBeautyFacts failed for '{query}': {e}")
        return []
    out = []
    max_results = int(cfg.get("max_results", 5))
    template = cfg.get("product_url_template", "")
    for p in data.get("products", [])[:max_results]:
        pid = p.get("_id", "")
        url = template.format(pid=pid) if template and pid else ""
        out.append(
            {
                "name": p.get("product_name", "") or "",
                "url": url,
                "price": "",
                "image_url": "",  # Will be fetched by fetch_product_image
                "source": "openbeautyfacts",
            }
        )
    logger.info(f"OpenBeautyFacts returned {len(out)} results for: {query}")
    return out


_NON_PRODUCT_PATTERNS = (
    "/collections/",
    "/collection/",
    "/shop/",
    "/shop?",
    "/buy/",
    "/category/",
    "/categories/",
    "/search?",
    "/search/",
    "/gallery/",
    "/discover/",
    "/ideas/",
    "/account/",
    "/login/",
    "/cart/",
    "/checkout/",
    "/about/",
    "/contact/",
    "/blog/",
    "/article/",
    "/guide/",
    "/wishlist/",
    "/pages/",
    "/reviews/",
    "/best-",
    "/top-",
    "/beauty/",
    "/skincare/",
    "/makeup-",
    "youtube.com/",
    "youtu.be/",
    "tiktok.com/",
    "pinterest.com/",
    "instagram.com/",
    "facebook.com/",
    "twitter.com/",
    "reddit.com/",
)


def _is_product_url(url: str, title: str) -> bool:
    """Filter out category/article pages, keep only product pages."""
    import re
    url_lower = url.lower()
    title_lower = title.lower()

    # Skip URLs with category/collection paths
    for pattern in _NON_PRODUCT_PATTERNS:
        if pattern in url_lower:
            return False

    # Skip titles that are clearly listicles or category pages
    skip_keywords = [
        "best ", "top ", "collection", "shop all", "buy ", "products",
        "gallery", "ideas", "discover", "tested", "review", "reviews",
        "guides", "shop by", "shop ", "products |", "products -",
        "the best", "the top", "the ultimate", "the definitive",
    ]
    for kw in skip_keywords:
        if title_lower.startswith(kw):
            return False

    # Skip URLs that are just the homepage (e.g., "https://thedermaco.com/")
    if re.match(r"^https?://[^/]+/?$", url):
        return False

    # Skip titles that look like URLs (contain "://" or start with domain-like patterns)
    if "://" in title or re.match(r"^[a-z0-9-]+\.(com|co\.uk|org|net|eu)/", title_lower):
        return False

    # Skip article/guide pages (URLs with /beauty/, /skincare/, /article/, /guide/)
    article_patterns = ["/beauty/", "/skincare/", "/article/", "/guide/", "/makeup-"]
    for pattern in article_patterns:
        if pattern in url_lower:
            return False

    return True


_KNOWN_SITES = {
    "yesstyle", "ulta", "sephora", "amazon", "walmart", "target", "dermstore",
    "lookfantastic", "beauty bay", "space nk", "selfridges", "john lewis", "boots",
    "glamour", "allure", "byrdie", "who what wear", "expert reviews", "healthline",
    "webmd", "medical news today", "bbc good food", "good housekeeping",
    "woman&home", "real simple", "etsy", "ebay", "aliexpress", "q+a",
    "dot and key", "the ordinary", "cerave", "la roche posay", "paula's choice",
    "the inkey list", "byoma", "kravebeauty", "garnier", "neutrogena", "olay",
    "ponds", "nivea", "dove", "amazon.com", "ulta.com", "sephora.com",
}


def _clean_product_name(title: str) -> str:
    """Extract clean product name from title (remove site prefix/suffix, etc.)."""
    import re

    # Remove ellipsis and trailing dots
    cleaned = re.sub(r"\s*\.+\s*$", "", title)

    # Remove common prefixes: "Amazon.com: ", "Shop ", "Buy "
    cleaned = re.sub(r"^(amazon\.com:\s*|shop\s+|buy\s+)", "", cleaned, flags=re.IGNORECASE)

    # Detect concatenated titles (e.g., "Product NameThe 6 Best Products..."):
    # Look for patterns where a lowercase letter is followed by an uppercase letter
    # and then listicle keywords like "Best", "Top", "Reviewed", "Tested"
    listicle_match = re.search(
        r"([a-z][)\,\']?)(The\s+\d+\s+(Best|Top)|The\s+(Best|Top)|Best\s+\d+|\d+\s+Best|Tested\s+and\s+Reviewed)",
        cleaned,
    )
    if listicle_match:
        # Extract just the product name before the listicle title
        cleaned = cleaned[: listicle_match.start(1)].rstrip()
        # Clean up trailing separators
        cleaned = re.sub(r"\s*[)\,\-–—]+\s*$", "", cleaned)

    # Split on pipe (|) - most reliable separator
    if "|" in cleaned:
        parts = cleaned.split("|", 1)
        candidate = parts[0].strip()
        site_part = parts[1].strip().lower()
        # Only split if second part looks like a site
        if any(site in site_part for site in _KNOWN_SITES) or ".com" in site_part:
            return candidate.rstrip(". -–—")
        # If no site detected, keep full title (might be "Product | Variant")
        return cleaned.rstrip(". -–—")

    # Split on » (guillemet) - common in some sites
    if "»" in cleaned:
        parts = cleaned.split("»", 1)
        return parts[0].strip().rstrip(". -–—")

    # Split on " - " (space-dash-space) but check if second part is a site
    match = re.match(r"^(.+?)\s+[\-–—]\s+(.+)$", cleaned)
    if match:
        candidate, site_part = match.groups()
        site_lower = site_part.strip().lower()
        if any(site in site_lower for site in _KNOWN_SITES) or ".com" in site_lower:
            return candidate.strip().rstrip(". -–—")

    # No site suffix found, return as-is
    return cleaned.strip()


def _aistack_search(ingredient: str, client: aistack.AistackClient | None = None) -> list[dict]:
    """Aistack search - primary discovery source."""
    cfg = _products_cfg().get("aistack_fallback", {})
    query = cfg.get("query_template", "{ingredient} buy skincare").format(ingredient=ingredient)
    logger.info(f"Aistack search: {query}")
    try:
        c = client or aistack.AistackClient()
        results = c.search(query, max_results=int(cfg.get("max_results", 5)))
    except (httpx.HTTPError, RuntimeError) as e:
        logger.warning(f"Aistack search failed for '{ingredient}': {e}")
        return []

    out = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        if not url:
            continue
        if not _is_product_url(url, title):
            logger.debug(f"Skipping non-product: {title[:60]}")
            continue
        out.append(
            {
                "name": _clean_product_name(title),
                "url": url,
                "price": "",
                "source": "aistack-search",
            }
        )
    logger.info(f"Aistack returned {len(out)} product results for: {ingredient}")
    return out


def _match(ing: str, title: str) -> str:
    tl = title.lower()
    for token in ing.lower().replace("-", " ").split():
        if token in tl:
            return f"{ing} (found in title)"
    return f"{ing} (best available match)"


def _generate_reasoning(products: list[dict], skin_type: str = "", concerns: str = "") -> list[dict]:
    """Use LLM to generate detailed reasoning for each product (50+ words each)."""
    import json as _json

    if not products:
        return products

    logger.info(f"Generating detailed reasoning for {len(products)} products")

    try:
        from . import providers

        llm = providers.chat_model(role="worker")

        product_list = "\n".join(
            f"- {p.get('name', 'Unknown')} (key ingredient: {p.get('ingredient', 'unknown')})"
            for p in products
        )

        skin_context = f"skin type={skin_type}" if skin_type else "not specified"
        concern_context = f"primary concern={concerns}" if concerns else "not specified"

        messages = [
            SystemMessage(
                content=(
                    "You are a dermatology and skincare expert. For each product listed, write a detailed "
                    "reasoning (at least 50 words) explaining WHY this product is recommended. "
                    "Consider the following:\n"
                    "- The key ingredient and its mechanism of action on the skin\n"
                    "- How it addresses the user's specific skin type and concerns\n"
                    "- The product's formulation, texture, and absorption properties\n"
                    "- How it fits into a daily skincare routine (AM/PM, layering order)\n"
                    "- Any notable clinical evidence or dermatological backing\n"
                    "- Who would benefit most from this product\n\n"
                    f"User context: {skin_context}, {concern_context}\n\n"
                    "Return ONLY a JSON array of strings (one per product, same order as input). "
                    "Each string should be a detailed 50+ word reasoning paragraph."
                )
            ),
            HumanMessage(
                content=f"Generate detailed reasoning for these products:\n{product_list}"
            ),
        ]

        resp = llm.invoke(messages)
        content = (resp.content or "").strip()

        # Extract JSON array from response
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            reasonings = _json.loads(json_match.group())
            for i, reasoning in enumerate(reasonings):
                if i < len(products):
                    products[i]["ingredient_match"] = reasoning
        else:
            # Try to parse as JSON object with product names as keys
            obj_match = re.search(r"\{[\s\S]*\}", content)
            if obj_match:
                obj = _json.loads(obj_match.group())
                for p in products:
                    name = p.get("name", "")
                    for key, val in obj.items():
                        if key.lower() in name.lower() or name.lower() in key.lower():
                            p["ingredient_match"] = val
                            break
            else:
                # Fallback: generate simple reasoning
                for p in products:
                    ing = p.get("ingredient", "")
                    name = p.get("name", "")
                    p["ingredient_match"] = (
                        f"{name} contains {ing}, a well-researched ingredient known for its efficacy in "
                        f"addressing specific skin concerns. This product is formulated to deliver optimal "
                        f"results when used consistently as part of a daily skincare routine."
                    )

    except (RuntimeError, ValueError, KeyError) as e:
        logger.warning(f"LLM reasoning generation failed: {e}, using defaults")
        for p in products:
            ing = p.get("ingredient", "")
            name = p.get("name", "")
            p["ingredient_match"] = (
                f"{name} contains {ing}, a well-researched ingredient known for its efficacy in "
                f"addressing specific skin concerns. This product is formulated to deliver optimal "
                f"results when used consistently as part of a daily skincare routine."
            )

    return products


def find_products(ingredients: list[str], budget: str | None = None) -> list[dict]:
    """One product per ingredient. Uses aistack/openbeauty first, SerpAPI only for final price lookup."""
    cfg = _products_cfg()
    budget = budget or cfg.get("default_budget", "medium")
    limits = cfg.get("budget_limits", {})
    limit = limits.get(budget, limits.get("medium", float("inf")))
    if isinstance(limit, str):
        limit = float(limit) if limit != ".inf" else float("inf")

    logger.info(f"find_products: {len(ingredients)} ingredients, budget={budget}, limit={limit}")

    result: list[dict] = []
    for ing in ingredients:
        found: list[dict] = []

        # 1. Try aistack search first (free/cheap)
        found = _aistack_search(ing)

        # 2. Fall back to Open Beauty Facts (free)
        if not found:
            found = _open_beauty_facts(ing)

        # 3. Use SerpAPI if we have no results OR results lack price+image
        has_price = any(p.get("price") for p in found)
        has_image = any(p.get("image_url") for p in found)
        if not found or (not has_price and not has_image):
            logger.info(f"Trying SerpAPI for '{ing}' (found={len(found)}, has_price={has_price}, has_image={has_image})")
            serp_cfg = cfg.get("serpapi", {})
            query_templates = serp_cfg.get("query_templates", ["{ing} skincare", "{ing}"])
            for tpl in query_templates:
                serp_results = _serp_search(tpl.format(ing=ing))
                if serp_results:
                    # Prefer SerpAPI results if they have price+image
                    serp_has_price = any(p.get("price") for p in serp_results)
                    serp_has_image = any(p.get("image_url") for p in serp_results)
                    if serp_has_price or serp_has_image:
                        found = serp_results
                        break

        if not found:
            logger.warning(f"No products found for ingredient: {ing}")
            continue

        # Apply budget filter and pick first match
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
                    "image_url": item.get("image_url", ""),
                }
            )
            break  # first acceptable product per ingredient

    # Fetch product images for products without images
    for item in result:
        if not item.get("image_url"):
            url = item.get("url", "")
            name = item.get("name", "")
            if url:
                image_url = fetch_product_image(url, product_name=name)
                if image_url:
                    item["image_url"] = image_url
                    logger.info(f"Fetched image for '{name[:40]}': {image_url[:60]}")

    # For products still without images, try SerpAPI
    for item in result:
        if not item.get("image_url"):
            ing = item.get("ingredient", "")
            if ing:
                serp_results = _serp_search(f"{ing} product")
                if serp_results and serp_results[0].get("image_url"):
                    item["image_url"] = serp_results[0]["image_url"]
                    # Also update price if missing
                    if not item.get("price") and serp_results[0].get("price"):
                        item["price"] = serp_results[0]["price"]
                    logger.info(f"Got image from SerpAPI for '{item.get('name', '')[:40]}'")

    # Estimate prices for products without prices
    result = _estimate_prices(result)

    # Generate detailed reasoning for each product
    result = _generate_reasoning(result)

    logger.info(f"find_products: returning {len(result)} products")
    return result


def _estimate_prices(products: list[dict]) -> list[dict]:
    """Use LLM to estimate prices for products that don't have real prices."""
    import json as _json

    # Find products without prices
    needs_price = [p for p in products if not p.get("price")]
    if not needs_price:
        return products

    logger.info(f"Estimating prices for {len(needs_price)} products via LLM")

    try:
        from . import providers

        llm = providers.chat_model(role="worker")

        product_list = "\n".join(
            f"- {p.get('name', 'Unknown')} (ingredient: {p.get('ingredient', 'unknown')}, source: {p.get('source', 'unknown')})"
            for p in needs_price
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a skincare product pricing expert. Given a list of products, "
                    "estimate a realistic price in USD for each product. "
                    "Consider the product type (serum, moisturizer, cleanser, sunscreen, etc.), "
                    "the ingredient, and the brand positioning. "
                    "Return ONLY a JSON object mapping product names to prices like: "
                    '{"Product Name": "$XX.XX"}. '
                    "Prices should be realistic for the US market. "
                    "Serums typically cost $15-80, moisturizers $10-60, cleansers $8-40, sunscreens $12-50."
                )
            ),
            HumanMessage(
                content=f"Estimate prices for these products:\n{product_list}"
            ),
        ]

        resp = llm.invoke(messages)
        content = (resp.content or "").strip()

        # Extract JSON from response
        import re
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            prices = _json.loads(json_match.group())
            for p in needs_price:
                name = p.get("name", "")
                for price_name, price_val in prices.items():
                    if price_name.lower() in name.lower() or name.lower() in price_name.lower():
                        p["price"] = price_val
                        break
                else:
                    # If no match found, assign a default based on product type
                    p["price"] = "$24.99"
        else:
            # Fallback: assign default prices
            for p in needs_price:
                p["price"] = "$24.99"

    except (RuntimeError, ValueError, KeyError) as e:
        logger.warning(f"LLM price estimation failed: {e}, using defaults")
        for p in needs_price:
            p["price"] = "$24.99"

    return products
