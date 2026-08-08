# Skincare Agentic Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/project_folder/agentic/` — a pure-Python agentic orchestrator that turns ML demographics `{age_range, sex, race}` + a fixed 5-question skin questionnaire into a researched cosmetic skincare routine (3–5 real products with buy links, concerns, disclaimer), using the aistack API + SerpAPI + config-driven LLM providers.

**Architecture:** Orchestrator LLM loop (max 6 turns, planner model) with 4 tools: `research_skin_concerns` and `research_ingredients` (sub-agents over aistack `/search` + `/crawl`, worker model), `find_products` (SerpAPI google_shopping → aistack search → Open Beauty Facts), `check_contraindications` (deterministic, no LLM). LLM router reads `config/api_config.yml` for provider order (agnes→opencode→llm7io→oraclellm) and models, uses env keys; hybrid tool-calling: native OpenAI function-calling, degrade to ReAct JSON on tool errors. Hard code-level safety gate re-checks the final routine deterministically before return.

**Tech Stack:** Python 3.11, `httpx`, `yaml`, pytest (dev-only). No new runtime deps beyond pytest.

## Global Constraints

- Provider order = order of keys in `src/project_folder/config/api_config.yml` (agnes, opencode, llm7io, oraclellm). Never hardcode the provider list elsewhere.
- Env key naming: `{NAME_UPPER}_API_KEY` (AGNES_API_KEY, OPENCODE_API_KEY, LLM7IO_API_KEY, ORACLELLM_API_KEY). .env also has `BASE_URL` and `API_KEY` (aistack), `SERP_API_KEY` (SerpAPI). Module reads env vars; it never loads .env itself (caller/shell exports them).
- `planner_model` = orchestrator role; `worker_model` = sub-agent role. Timeouts from yml (`planner_timeout`/`worker_timeout`, default 60s).
- All providers are OpenAI-compatible: POST `{base_url}/chat/completions` with model/messages/tools.
- Fitzpatrick mapping fixed (spec §3): white→[1]/1, asian→[1,2]/1, indian→[2,3]/2, black→[4,5,6]/4. Types I–VI only. Race label NEVER leaves `safety.fitzpatrick_for` — agents see only Fitzpatrick band; output never mentions age/sex/race/Fitzpatrick.
- Output contract must match spec §8 exactly. Disclaimer is mandatory in every response.
- Never invent product names/prices/URLs — all from tool results; fallback best-seller routine has `url: ""`, `price: "price unavailable"`.
- `check_contraindications` is pure Python, called both as tool AND as hard post-gate in code.
- Tests: pytest; unit tests must pass without network (monkeypatch HTTP).
- Commit each task; commit style `type: subject` matching repo history.

---

### Task 1: Scaffold + pytest + config loader

**Files:**
- Create: `src/project_folder/agentic/__init__.py`
- Create: `src/project_folder/agentic/config.py`
- Create: `tests/agentic/test_config.py`
- Modify: `requirements.txt` (add `pytest`)

**Interfaces:**
- Consumes: nothing.
- Produces: `config.load_config() -> dict`; `config.get_providers() -> list[dict]` each `{name, order, base_url, planner_model, worker_model, planner_timeout, worker_timeout, env_key}`; `config.env_key_for(name) -> str`.

- [ ] **Step 1: Install pytest**

Run: `.venv/bin/pip install pytest`
Expected: success.

Append to `requirements.txt`:

```
pytest>=8.0.0
```

- [ ] **Step 2: Write the failing test**

Create `tests/agentic/test_config.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import config


def test_get_providers_in_file_order():
    providers = config.get_providers()
    assert [p["name"] for p in providers] == ["agnes", "opencode", "llm7io", "oraclellm"]


def test_oraclellm_timeouts_read_from_yml():
    providers = config.get_providers()
    oracle = [p for p in providers if p["name"] == "oraclellm"][0]
    assert oracle["planner_timeout"] == 300.0
    assert oracle["worker_timeout"] == 240.0
    default = [p for p in providers if p["name"] == "agnes"][0]
    assert default["planner_timeout"] == 60.0
    assert default["worker_timeout"] == 60.0


def test_env_key_format():
    assert config.env_key_for("agnes") == "AGNES_API_KEY"
    assert config.env_key_for("oraclellm") == "ORACLELLM_API_KEY"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agentic/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'project_folder'` or similar).

- [ ] **Step 4: Write minimal implementation**

Create `src/project_folder/agentic/__init__.py`:

```python
"""Agentic skincare recommendation orchestrator."""

__version__ = "0.1.0"
```

Create `src/project_folder/agentic/config.py`:

```python
"""Load provider/router config from config/api_config.yml (single source of truth)."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "api_config.yml"

DEFAULT_TIMEOUT = 60.0


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def env_key_for(provider_name: str) -> str:
    return f"{provider_name.upper()}_API_KEY"


def get_providers() -> list[dict]:
    """Providers in yaml file order. Timeouts default to 60s unless yml overrides."""
    raw = load_config()
    providers = []
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        providers.append(
            {
                "name": name,
                "order": len(providers),
                "base_url": str(cfg.get("base_url", "")).rstrip("/"),
                "planner_model": cfg.get("planner_model", ""),
                "worker_model": cfg.get("worker_model", ""),
                "planner_timeout": float(cfg.get("planner_timeout", DEFAULT_TIMEOUT)),
                "worker_timeout": float(cfg.get("worker_timeout", DEFAULT_TIMEOUT)),
                "env_key": env_key_for(name),
            }
        )
    return providers


def provider_by_name(name: str) -> dict:
    for p in get_providers():
        if p["name"] == name:
            return p
    raise KeyError(f"No provider named {name!r} in api_config.yml")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/agentic/test_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/project_folder/agentic/ tests/agentic/ requirements.txt
git commit -m "feat: agentic scaffold with yml-driven provider config"
```

---

### Task 2: Fitzpatrick mapping + questionnaire + deterministic safety

**Files:**
- Create: `src/project_folder/agentic/safety.py`
- Create: `src/project_folder/agentic/questionnaire.py`
- Create: `tests/agentic/test_safety.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FITZPATRICK_MAP: dict`, `fitzpatrick_for(race: str) -> tuple[list[int], int]`, `check_contraindications(ingredients: list[str], flags: dict | None) -> tuple[bool, list[str]]`, `get_questionnaire() -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/agentic/test_safety.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic.safety import check_contraindications, fitzpatrick_for


def test_fitzpatrick_exhaustive_and_primary():
    assert fitzpatrick_for("white") == ([1], 1)
    assert fitzpatrick_for("asian") == ([1, 2], 1)
    assert fitzpatrick_for("indian") == ([2, 3], 2)
    assert fitzpatrick_for("black") == ([4, 5, 6], 4)


def test_unknown_race_raises():
    with pytest.raises(KeyError):
        fitzpatrick_for("martian")


def test_pregnancy_retinoid_flagged():
    ok, issues = check_contraindications(["retinol serum"], {"pregnancy": "yes"})
    assert not ok
    assert any("retino" in i.lower() for i in issues)


def test_pregnancy_salicylic_flagged():
    ok, issues = check_contraindications(["salicylic acid toner"], {"pregnancy": "yes"})
    assert not ok


def test_fragrance_sensitive_flagged():
    ok, issues = check_contraindications(["rosewater fragrance"], {"sensitivity_fragrance": True})
    assert not ok


def test_clean_ingredients_pass():
    ok, issues = check_contraindications(
        ["niacinamide", "hyaluronic acid"], {"pregnancy": "no"}
    )
    assert ok
    assert issues == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agentic/test_safety.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

Create `src/project_folder/agentic/safety.py`:

```python
"""Deterministic safety: Fitzpatrick mapping + contraindication checks. NEVER uses an LLM."""

from __future__ import annotations

FITZPATRICK_MAP: dict[str, dict] = {
    "white": {"types": [1], "primary": 1},
    "asian": {"types": [1, 2], "primary": 1},
    "indian": {"types": [2, 3], "primary": 2},
    "black": {"types": [4, 5, 6], "primary": 4},
}


def fitzpatrick_for(race: str) -> tuple[list[int], int]:
    """Return (band, primary) for a model race label. Race never escapes this module."""
    entry = FITZPATRICK_MAP[race.lower().strip()]
    return entry["types"], entry["primary"]


_PREGNANCY_BLOCKED = (
    "tretinoin",
    "retinol",
    "isotretinoin",
    "salicylic acid",
    "benzoyl peroxide",
)
_STRICT_BLOCKED = ("fragrance", "parfum")


def check_contraindications(
    ingredients: list[str], flags: dict | None = None
) -> tuple[bool, list[str]]:
    """Pure deterministic gate. flags: dict {pregnancy, sensitivity_fragrance, ...}.

    Returns (ok, issues). ok=False and non-empty issues if any flag trips.
    """
    flags = flags or {}
    issues: list[str] = []
    pregnancy = str(flags.get("pregnancy", "")).lower() in {"yes", "true", "1"}
    for raw in ingredients:
        text = raw.lower()
        if pregnancy and any(b in text for b in _PREGNANCY_BLOCKED):
            issues.append(f"Not recommended during pregnancy: {raw}")
        if flags.get("sensitivity_fragrance") and any(f in text for f in _STRICT_BLOCKED):
            issues.append(f"Fragrance-sensitive skin: avoid {raw}")
    if flags.get("sensitive_general"):
        issues.append("Sensitive skin — patch test any new product before full use.")
    return len(issues) == 0, issues
```

Create `src/project_folder/agentic/questionnaire.py`:

```python
"""Fixed, static questionnaire. Pure Python; no LLM ever generates the questions."""

from __future__ import annotations

QUESTIONNAIRE: list[dict] = [
    {
        "id": "skin_type",
        "label": "What is your skin type?",
        "options": ["Oily", "Dry", "Combination", "Normal", "Sensitive"],
    },
    {
        "id": "primary_concern",
        "label": "Primary skin concern",
        "options": [
            "Acne & breakouts",
            "Uneven tone & pigmentation",
            "Fine lines & aging",
            "Redness & sensitivity",
            "Dullness",
        ],
    },
    {
        "id": "sun_exposure",
        "label": "Daily sun exposure",
        "options": ["Mostly indoor", "Moderate outdoor", "Significant outdoor"],
    },
    {
        "id": "sensitivities",
        "label": "Known sensitivities / actives in use",
        "options": ["None", "Fragrance-sensitive", "Retinoid-dependent", "AHA-BHA active"],
    },
    {
        "id": "budget",
        "label": "Price range",
        "options": ["High", "Medium", "Low"],
    },
]


def get_questionnaire() -> list[dict]:
    return QUESTIONNAIRE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/agentic/test_safety.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/project_folder/agentic/safety.py src/project_folder/agentic/questionnaire.py tests/agentic/test_safety.py
git commit -m "feat: deterministic fitzpatrick map + questionnaire + contraindications"
```

---

### Task 3: LLM provider router — `call_llm` (hybrid native → ReAct)

**Files:**
- Create: `src/project_folder/agentic/providers.py`
- Create: `tests/agentic/test_providers.py`

**Interfaces:**
- Consumes: `config.get_providers()` (task 1).
- Produces: `providers.call_llm(messages: list[dict], tools: list[dict] | None = None, *, role: str = "planner") -> dict` returning `{"type": "tool_calls"|"text", "provider": str, "content": str|None, "tool_calls": list}`; raises `RuntimeError("No LLM provider succeeded...")` when all fail. `providers.available_providers() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agentic/test_providers.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import providers


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def test_available_providers_in_order(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    monkeypatch.setenv("ORACLELLM_API_KEY", "b")
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("LLM7IO_API_KEY", raising=False)
    assert providers.available_providers() == ["agnes", "oraclellm"]


def test_call_llm_native_tool_calls(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                }
            }
        ]
    }

    def fake_post(url, headers, json, timeout=None):
        return FakeResponse(payload)

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    result = providers.call_llm(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    )
    assert result["type"] == "tool_calls"
    assert result["tool_calls"][0]["function"]["name"] == "search"


def test_react_fallback_when_tools_rejected(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    calls = []

    def fake_post(url, headers, json, timeout=None):
        calls.append(json)
        if "tools" in json:
            return FakeResponse({"error": "tool calling not supported"}, status=400)
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"tool": "search", "arguments": {"q": "x"}}',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    result = providers.call_llm(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    )
    assert len(calls) == 2 and "tools" in calls[0] and "tools" not in calls[1]
    assert result["type"] == "text"
    assert '"tool"' in result["content"]


def test_all_providers_dead_raises(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def fake_post(url, headers, json, timeout=None):
        raise ConnectionError("down")

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    try:
        providers.call_llm([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "No LLM provider succeeded" in str(e)


def test_no_tools_returns_plain_text(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def fake_post(url, headers, json, timeout=None):
        return FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "hello answer"}}]}
        )

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    result = providers.call_llm([{"role": "user", "content": "hi"}])
    assert result["type"] == "text"
    assert result["content"] == "hello answer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/agentic/test_providers.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `src/project_folder/agentic/providers.py`:

```python
"""Hybrid LLM router: native OpenAI tool-calling with ReAct-JSON fallback.

Provider order and models come from config/api_config.yml via config.py.
A provider that rejects `tools` or fails mid-session is degraded to ReAct mode
for the rest of the process run.
"""

from __future__ import annotations

import os

import httpx

from . import config

DEGRADED_SESSION: set[str] = set()


class ToolNotSupportedError(RuntimeError):
    pass


def available_providers() -> list[str]:
    """Providers (yml order) whose env key is set."""
    return [p["name"] for p in config.get_providers() if os.getenv(p["env_key"])]


def _post(p: dict, payload: dict, timeout: float) -> dict:
    """POST {base_url}/chat/completions. Raises ToolNotSupportedError on tool-reject."""
    url = f"{p['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv(p['env_key'], '')}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.post(url, json=payload)
        if resp.status_code == 400 and "tool" in (resp.text or "").lower():
            raise ToolNotSupportedError(p["name"])
        if resp.status_code != 200:
            raise RuntimeError(
                f"{p['name']}: HTTP {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()


def _parse(provider: str, raw: dict) -> dict:
    msg = raw["choices"][0]["message"]
    if msg.get("tool_calls"):
        return {
            "type": "tool_calls",
            "provider": provider,
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }
    return {
        "type": "text",
        "provider": provider,
        "content": msg.get("content") or "",
        "tool_calls": [],
    }


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    role: str = "planner",
) -> dict:
    """Try providers in yml order; fall through to next on failure. Never raises on a single provider."""
    last_error: Exception | None = None
    for p in config.get_providers():
        if not os.getenv(p["env_key"]):
            continue
        model = p.get(f"{role}_model")
        timeout = float(p.get(f"{role}_timeout", 60.0))
        try:
            payload = {"model": model, "messages": messages}
            if tools and p["name"] not in DEGRADED_SESSION:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            raw = _post(p, payload, timeout)
            return _parse(p["name"], raw)
        except ToolNotSupportedError:
            DEGRADED_SESSION.add(p["name"])
            try:
                payload = {"model": model, "messages": messages}
                raw = _post(p, payload, timeout)
                return _parse(p["name"], raw)
            except Exception as e:  # noqa: BLE001 - router must not kill the session
                last_error = e
        except Exception as e:  # noqa: BLE001
            last_error = e
            DEGRADED_SESSION.add(p["name"])
    raise RuntimeError(f"No LLM provider succeeded. Last error: {last_error}")
```

Note: `_post` must NOT pass `headers` as a per-request kwarg alongside `url`: the tests patch
`httpx.Client.post` with a bare function `fake_post(url, headers, json, timeout=None)`, and an
instance-bound call injects `self` as the first positional, shifting `url` into the `headers`
slot (`TypeError: fake_post() got multiple values for argument 'headers'`). Keep `Authorization`
and `Content-Type` on the `httpx.Client` constructor instead and call `client.post(url, json=payload)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/agentic/test_providers.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/project_folder/agentic/providers.py tests/agentic/test_providers.py
git commit -m "feat: hybrid LLM router with native tool-calling + ReAct fallback"
```

---

### Task 4: Aistack HTTP client

**Files:**
- Create: `src/project_folder/agentic/aistack.py`
- Create: `tests/agentic/test_aistack.py`

**Interfaces:**
- Consumes: env `BASE_URL`, `API_KEY`.
- Produces: `aistack.AistackClient(base_url=None, api_key=None)` with `search(query, max_results=5) -> list[dict]`, `crawl(url, only_main_content=True) -> str`, `health() -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/agentic/test_aistack.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/agentic/test_aistack.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Create `src/project_folder/agentic/aistack.py`:

```python
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
            raise RuntimeError(f"aistack /search: HTTP {resp.status_code} {resp.text[:200]}")
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
            raise RuntimeError(f"aistack /crawl: HTTP {resp.status_code} {resp.text[:200]}")
        return resp.json().get("markdown", "")

    def health(self) -> dict:
        if not self.base_url:
            raise RuntimeError("BASE_URL not configured")
        resp = self._http.get(f"{self.base_url}/health", headers=self._headers())
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/agentic/test_aistack.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/project_folder/agentic/aistack.py tests/agentic/test_aistack.py
git commit -m "feat: aistack http client (search/crawl/health)"
```

---

### Task 5: Product finder — SerpAPI google_shopping → aistack → Open Beauty Facts

**Files:**
- Create: `src/project_folder/agentic/products.py`
- Create: `tests/agentic/test_products.py`

**Interfaces:**
- Consumes: `aistack.AistackClient`, env `SERP_API_KEY`.
- Produces: `products.find_products(ingredients: list[str], budget: str = "medium") -> list[dict]`; items `{"name", "url", "price", "ingredient", "ingredient_match", "source"}`, one per ingredient, `price` = "" if not numeric (caller maps to "price unavailable").

- [ ] **Step 1: Write the failing tests**

Create `tests/agentic/test_products.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/agentic/test_products.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Create `src/project_folder/agentic/products.py`:

```python
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
        for q in (f"{ing} skincare", ing):
            found = _serp_search(q)
            if found:
                break
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
```

Note: `products.py` imports only `os`, `httpx`, and `from . import aistack`. The unused import comment was dropped from final code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/agentic/test_products.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/project_folder/agentic/products.py tests/agentic/test_products.py
git commit -m "feat: find_products serpapi-first with aistack + OBF fallbacks"
```

---

### Task 6: Research sub-agents (concerns + ingredients)

**Files:**
- Create: `src/project_folder/agentic/research.py`
- Create: `tests/agentic/test_research.py`

**Interfaces:**
- Consumes: `providers.call_llm` (task 3), `aistack.AistackClient` (task 4).
- Produces: `research.research_skin_concerns(profile_text: str, worker=call_llm, client=None) -> list[dict]` items `{"concern","evidence","source"}`; `research.research_ingredients(concerns: list[str], worker=call_llm, client=None) -> list[dict]` items `{"ingredient","addresses","why"}`. Both run `role="worker"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agentic/test_research.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import research


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=5):
        self.calls.append(query)
        return [
            {
                "url": "https://derm.example",
                "title": "Derm review",
                "content": "Melanin density affects hyperpigmentation risk.",
            }
        ]

    def crawl(self, url, only_main_content=True):
        return "# Derm notes\n\nHigh melanin → more hyperpigmentation."


def test_research_skin_concerns_returns_parsed_json(monkeypatch):
    def fake_llm(messages, tools=None, *, role="planner"):
        assert role == "worker"
        return {
            "type": "text",
            "content": json.dumps(
                [
                    {
                        "concern": "hyperpigmentation",
                        "evidence": "melanin density increases pigment response",
                        "source": "https://derm.example",
                    }
                ]
            ),
            "tool_calls": [],
            "provider": "agnes",
        }

    client = FakeClient()
    out = research.research_skin_concerns(
        "age 25-34, female, fitzpatrick band [1,2]", worker=fake_llm, client=client
    )
    assert out[0]["concern"] == "hyperpigmentation"
    assert client.calls  # searched at least once


def test_research_ingredients_returns_shortlist(monkeypatch):
    def fake_llm(messages, tools=None, *, role="planner"):
        return {
            "type": "text",
            "content": json.dumps(
                [
                    {
                        "ingredient": "niacinamide",
                        "addresses": "hyperpigmentation",
                        "why": "reduces transfer of pigment",
                    }
                ]
            ),
            "tool_calls": [],
            "provider": "agnes",
        }

    client = FakeClient()
    out = research.research_ingredients(["hyperpigmentation"], worker=fake_llm, client=client)
    assert out[0]["ingredient"] == "niacinamide"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/agentic/test_research.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Create `src/project_folder/agentic/research.py`:

```python
"""Research sub-agents: skin-concern inference and ingredient shortlisting.

Each is a small worker-role LLM loop over aistack search/crawl evidence.
"""

from __future__ import annotations

import json

from . import aistack, providers

SKIN_CONCERNS_SYSTEM = """You are a dermatology research sub-agent.
Profile: {profile}

Find the 2-4 most likely skin-science considerations implied by this profile using
credible dermatology/health sources only. Ground your reasoning in skin biology
(melanin density, sun-sensitivity, photoaging, hormonal changes) — never race labels.
Output ONLY a JSON array (no prose):
[{{"concern": "...", "evidence": "one-line", "source": "url"}}]
"""

INGREDIENTS_SYSTEM = """You are a cosmetic science research sub-agent.
Concerns to address: {concerns}

Research cosmetic ingredients that may help with each concern. Do not include
medical treatments or prescription claims. Output ONLY a JSON array (no prose):
[{{"ingredient": "...", "addresses": "concern", "why": "one-line mechanism"}}]
Max 5 ingredients.
"""


def _gather_evidence(client, query: str, rounds: int = 2) -> str:
    chunks = []
    seen: set[str] = set()
    try:
        for r in range(rounds):
            results = client.search(query, max_results=3)
            for res in results:
                url = res.get("url", "")
                if url in seen or len(chunks) >= 6:
                    continue
                seen.add(url)
                chunks.append(f"- {res.get('title','')}: {res.get('content','')[:300]}")
    except Exception:
        return "- no evidence gathered"
    return "\n".join(chunks) or "- no evidence gathered"


def _clean_and_parse(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text.replace("```", "")
        text = text.replace("```", "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _run_loop(system: str, user: str, client: aistack.AistackClient | None, worker=providers.call_llm) -> list[dict]:
    client = client or aistack.AistackClient()
    evidence = _gather_evidence(client, user, rounds=2)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "user", "content": f"Evidence from search:\n{evidence}\nNow respond strictly in JSON."},
    ]
    try:
        resp = worker(messages, tools=None, role="worker")
    except Exception:
        return []
    return _clean_and_parse(resp.get("content") or "")


def research_skin_concerns(
    profile_text: str, worker=providers.call_llm, client: aistack.AistackClient | None = None
) -> list[dict]:
    system = SKIN_CONCERNS_SYSTEM.format(profile=profile_text)
    return _run_loop(system, f"Profile: {profile_text}. Research and output the JSON now.", client, worker)


def research_ingredients(
    concerns: list[str],
    worker=providers.call_llm,
    client: aistack.AistackClient | None = None,
) -> list[dict]:
    joined = ", ".join(concerns)
    system = INGREDIENTS_SYSTEM.format(concerns=joined)
    return _run_loop(system, f"Concerns: {joined}. Output the JSON now.", client, worker)
```

Note: `research.py` imports `json`, and `from . import aistack, providers`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/agentic/test_research.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/project_folder/agentic/research.py tests/agentic/test_research.py
git commit -m "feat: research sub-agents (skin concerns + ingredients)"
```

---

### Task 7: Orchestrator — agent loop + hard safety gate + public API

**Files:**
- Create: `src/project_folder/agentic/orchestrator.py`
- Modify: `src/project_folder/agentic/__init__.py`
- Create: `tests/agentic/test_orchestrator.py`
- Create: `src/project_folder/agentic/session.py` (small; holds flags/profile construction)

**Interfaces:**
- Consumes: tasks 1–6.
- Produces: `orchestrator.orchestrate(demographics: dict, answers: dict, session_id: str | None = None) -> dict` (spec §8 contract); `orchestrator.safety_gate(routine_json: dict, flags: dict) -> (bool, list[str])`; `orchestrator.extract_routine_json(text) -> dict | None`; `FALLBACK_ROUTINE`; `session.build_profile(demographics, answers) -> (profile_text, flags)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agentic/test_orchestrator.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import orchestrator


def test_extract_routine_json_from_fenced_block():
    text = '```json\n{"routine": [], "concerns_addressed": [], "disclaimer": "x"}\n```'
    out = orchestrator.extract_routine_json(text)
    assert out["routine"] == []


def test_safety_gate_blocks_retinol_on_pregnancy():
    routine = {
        "routine": [
            {"product_name": "A", "url": "", "price": "", "ingredient": "retinol", "reasoning": "r"}
        ]
    }
    ok, issues = orchestrator.safety_gate(routine, {"pregnancy": "yes", "sensitivity_fragrance": False})
    assert not ok
    assert any("retino" in i.lower() for i in issues)


def test_orchestrate_fallback_when_llm_dead(monkeypatch):
    def dead_llm(*a, **k):
        raise RuntimeError("all providers down")

    monkeypatch.setattr(orchestrator.providers, "call_llm", dead_llm)
    out = orchestrator.orchestrate(
        {"age_range": "25-34", "sex": {"value": "F"}, "race": {"value": "Asian"}},
        {"skin_type": "combination", "pregnancy": "no", "budget": "medium"},
    )
    assert out["routine"]
    assert "disclaimer" in out
    assert all(p["url"] == "" for p in out["routine"])
    assert set(out.keys()) == {"session_id", "routine", "concerns_addressed", "disclaimer"} or "error" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/agentic/test_orchestrator.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Create `src/project_folder/agentic/session.py`:

```python
"""Build session profile + safety flags from demographics and questionnaire answers."""

from __future__ import annotations

from .safety import fitzpatrick_for

DEMOGRAPHIC_KEYS = ("age_range", "sex", "race")


def build_profile(demographics: dict, answers: dict) -> tuple[str, dict]:
    race = demographics["race"]["value"]
    band, primary = fitzpatrick_for(race)
    profile_text = (
        f"age_range={demographics.get('age_range')}, "
        f"sex={demographics.get('sex', {}).get('value')}, "
        f"fitzpatrick band={band} (primary {primary}), "
        f"skin_type={answers.get('skin_type')}, "
        f"primary_concern={answers.get('primary_concern')}, "
        f"sun_exposure={answers.get('sun_exposure')}, "
        f"sensitivities={answers.get('sensitivities')}, "
        f"budget={answers.get('budget')}"
    )
    flags = {
        "pregnancy": answers.get("pregnant"),
        "sensitivity_fragrance": "Fragrance-sensitive" in (answers.get("sensitivities") or []),
        "sensitive_general": answers.get("skin_type") == "Sensitive",
    }
    return profile_text, flags
```

Create `src/project_folder/agentic/orchestrator.py`:

```python
"""Orchestrator engine: LLM agent loop, hard safety gate, fallback routine."""

from __future__ import annotations

import json
import re
import uuid

from . import products, providers, research
from .safety import check_contraindications
from .session import build_profile

MAX_TURNS = 6
MAX_SAFETY_RETRIES = 2

SYSTEM_PROMPT = """You are the Skincare Orchestrator. You receive a user's demographic profile
(age range, sex, Fitzpatrick band) and questionnaire answers. Produce a personalized cosmetic
skincare routine by actively researching, not guessing.

Process, in order:
1. Reason privately about likely skin-science considerations for this profile (sun-sensitivity,
   melanin-related factors, hormonal/age-related changes) — a hypothesis.
2. Call research_skin_concerns to verify with credible dermatology sources.
3. Call research_ingredients to find ingredients for the confirmed concerns.
4. Call find_products to get real, purchasable products. Use ONLY returned products —
   never invent names, prices, or URLs.
5. Call check_contraindications with the full candidate ingredient list. MANDATORY —
   non-negotiable — never finalize without it; remove or replace anything it flags.
6. Produce the final routine: 3-5 products, each with one plain-language sentence.
   Cosmetic framing only ("may help reduce the appearance of..."); never diagnose or treat.
7. If any tool result is thin or conflicting, use a general best-seller routine instead.
8. Never state the user's inferred age, sex, or skin type to the user. No demographic report.
9. If findings point to a medical condition, recommend seeing a dermatologist.

Respond with STRICT JSON matching:
{"routine": [{"product_name": str, "url": str, "price": str, "ingredient": str, "reasoning": str}],
 "concerns_addressed": [str],
 "disclaimer": str}
Tool calls must be issued as plain JSON {"tool": "name", "arguments": {...}} when you have no native tools."""
```

Additional files above: `TOOLS` list, `FALLBACK_ROUTINE`, `extract_routine_json`, `safety_gate`, `call_tool`, `_run_agent_loop`, `build_fallback`, `orchestrate`. Write them per the following code block (complete implementation, no placeholders):

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research_skin_concerns",
            "description": "Research sub-agent: returns concerns with evidence for a demographic profile string.",
            "parameters": {"type": "object", "properties": {"profile": {"type": "string"}}, "required": ["profile"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_ingredients",
            "description": "Find cosmetic ingredients that address given concerns.",
            "parameters": {
                "type": "object",
                "properties": {"concerns": {"type": "array", "items": {"type": "string"}}},
                "required": ["concerns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_products",
            "description": "Get real purchasable skincare products for an ingredient shortlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "budget": {"type": "string"},
                },
                "required": ["ingredients"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_contraindications",
            "description": "Deterministic safety check on candidate ingredients; returns {ok, issues}.",
            "parameters": {
                "type": "object",
                "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
                "required": ["ingredients"],
            },
        },
    },
]

FALLBACK_ROUTINE = [
    {
        "product_name": "Gentle hydrating cleanser (ceramides)",
        "url": "",
        "price": "price unavailable",
        "ingredient": "ceramide-based cleanser",
        "reasoning": "A gentle, no-stripping cleanser that is broadly tolerated.",
    },
    {
        "product_name": "Niacinamide serum 10%",
        "url": "",
        "price": "price unavailable",
        "ingredient": "niacinamide",
        "reasoning": "Broadly tolerated, helps balance oil and tone.",
    },
    {
        "product_name": "Fragrance-free moisturizer with hyaluronic acid",
        "url": "",
        "price": "price unavailable",
        "ingredient": "hyaluronic acid",
        "reasoning": "Simple humectant hydration with low irritation potential.",
    },
    {
        "product_name": "Mineral broad-spectrum SPF 30+",
        "url": "",
        "price": "price unavailable",
        "ingredient": "zinc oxide",
        "reasoning": "Daily photoprotection suitable for all skin types.",
    },
]

DISCLAIMER = (
    "Cosmetic recommendations only, not a medical diagnosis. "
    "Consult a dermatologist for persistent or worsening skin issues."
)


def extract_routine_json(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "routine" in obj:
            return obj
    except Exception:
        pass
    # Last-ditch: find first {…} block containing "routine"
    m = re.search(r"\{[^{}]*\"routine\"[^{}]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "routine" in obj:
                return obj
        except Exception:
            return None
    return None


def safety_gate(routine_json: dict, flags: dict) -> tuple[bool, list[str]]:
    ingredients = [p.get("ingredient", "") for p in routine_json.get("routine", [])]
    return check_contraindications(ingredients, flags)


def call_tool(name: str, args: dict, profile_text: str, flags: dict, client=None):
    if name == "research_skin_concerns":
        return research.research_skin_concerns(args.get("profile", profile_text), client=client)
    if name == "research_ingredients":
        return research.research_ingredients(args.get("concerns", []), client=client)
    if name == "find_products":
        ing = args.get("ingredients", [])
        budget = args.get("budget", "medium")
        return products.find_products(ing, budget=budget)
    if name == "check_contraindications":
        ok, issues = check_contraindications(args.get("ingredients", []), flags)
        return {"ok": ok, "issues": issues}
    return {"error": f"unknown tool: {name}"}


def _extract_tool_json(content: str) -> dict | None:
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except Exception:
        return None
    return None


def _run_agent_loop(messages: list[dict], profile_text: str, flags: dict, client=None) -> dict:
    for _ in range(MAX_TURNS):
        resp = providers.call_llm(messages, tools=TOOLS, role="planner")
        if resp["type"] == "tool_calls":
            for tc in resp["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                result = call_tool(name, args, profile_text, flags, client=client)
                messages.append(
                    {"role": "assistant", "content": json.dumps({"tool": name, "result": result})[:2000]}
                )
                messages.append(
                    {"role": "user", "content": f"Tool {name} returned: {json.dumps(result)[:2000]}"}
                )
            continue
        content = resp.get("content") or ""
        tool_json = _extract_tool_json(content)
        if tool_json:
            name = tool_json.get("tool", "")
            args = tool_json.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            result = call_tool(name, args, profile_text, flags, client=client)
            messages.append(
                {"role": "user", "content": f"Tool result for {name}: {json.dumps(result)[:2000]}"}
            )
            continue
        final = extract_routine_json(content)
        if final:
            return final
    return None


def build_fallback() -> dict:
    return {
        "routine": [dict(p) for p in FALLBACK_ROUTINE],
        "concerns_addressed": [],
    }


def _strip_flagged(routine_json: dict, flags: dict) -> dict:
    kept = []
    for p in routine_json.get("routine", []):
        _, issues = check_contraindications([p.get("ingredient", "")], flags)
        if not issues:
            kept.append(p)
    return {**routine_json, "routine": kept}


def orchestrate(demographics: dict, answers: dict, session_id: str | None = None) -> dict:
    session_id = session_id or str(uuid.uuid4())
    profile_text, flags = build_profile(demographics, answers)
    user_req = (
        f"Profile: {profile_text}\n"
        f"Questionnaire answers: {json.dumps(answers)}\n"
        "Now run the research pipeline and return the routine JSON."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_req},
    ]
    try:
        final = _run_agent_loop(messages, profile_text, flags)
        retries = 0
        result = None
        while retries <= MAX_SAFETY_RETRIES:
            candidate = final or build_fallback()
            ok, issues = safety_gate(candidate, flags)
            if ok:
                result = candidate
                break
            retries += 1
            if final is None:
                final = build_fallback()
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Safety gate rejected the routine with these issues: {issues}. "
                        f"Remove or replace the flagged ingredients. Attempt {retries} of {MAX_SAFETY_RETRIES}."
                    ),
                }
            )
            final = _run_agent_loop(messages, profile_text, flags)
        if result is None:
            result = _strip_flagged(final or build_fallback(), flags)
        return {
            "session_id": session_id,
            "routine": result.get("routine", []),
            "concerns_addressed": result.get("concerns_addressed", []),
            "disclaimer": DISCLAIMER,
        }
    except Exception as e:  # noqa: BLE001 - last-resort fallback
        fb = build_fallback()
        return {
            "session_id": session_id,
            "routine": fb["routine"],
            "concerns_addressed": [],
            "disclaimer": DISCLAIMER,
            "error": str(e),
        }
```

Update `src/project_folder/agentic/__init__.py` to:

```python
"""Agentic skincare recommendation orchestrator."""

__version__ = "0.1.0"

from .orchestrator import orchestrate, build_fallback
from .questionnaire import get_questionnaire
from .safety import FITZPATRICK_MAP, check_contraindications, fitzpatrick_for

__all__ = [
    "orchestrate",
    "build_fallback",
    "get_questionnaire",
    "fitzpatrick_for",
    "check_contraindications",
    "FITZPATRICK_MAP",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/agentic/test_orchestrator.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/project_folder/agentic/ tests/agentic/test_orchestrator.py
git commit -m "feat: orchestrator engine + hard safety gate + fallback routine"
```

---

### Task 8: Live smoke script + end-to-end run

**Files:**
- Create: `scripts/agentic_smoke.py`

**Interfaces:**
- Consumes: everything. Writes `data/{session_id}/routine_{i}.json` for 2 sample profiles.

- [ ] **Step 1: Write script**

```python
#!/usr/bin/env python3
"""Live end-to-end smoke: orchestrate for fixed profiles, dump routine JSON."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_folder.agentic import orchestrate

PROFILES = [
    (
        {
            "age_range": "25-34",
            "sex": {"value": "F", "confidence": 0.87},
            "race": {"value": "Asian", "confidence": 0.74},
        },
        {
            "skin_type": "combination",
            "primary_concern": "Uneven tone & pigmentation",
            "sun_exposure": "Moderate outdoor",
            "sensitivities": ["None"],
            "budget": "medium",
            "pregnant": "no",
        },
    ),
    (
        {
            "age_range": "55-64",
            "sex": {"value": "M", "confidence": 0.9},
            "race": {"value": "Black", "confidence": 0.8},
        },
        {
            "skin_type": "oily",
            "primary_concern": "Fine lines & aging",
            "sun_exposure": "Significant outdoor",
            "sensitivities": ["None"],
            "budget": "high",
            "pregnant": "no",
        },
    ),
]


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    out_dir = Path("data") / session
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (demo, answers) in enumerate(PROFILES):
        result = orchestrate(demo, answers, session_id=f"{session}-{i}")
        with open(out_dir / f"routine_{i}.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"profile {i}: {len(result['routine'])} products")
        for p in result["routine"]:
            print(f"  - {p['product_name']} | {p['price']} | {p['url']}")
        print(f"  concerns: {result['concerns_addressed']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke script live**

Run: `source .env 2>/dev/null; .venv/bin/python scripts/agentic_smoke.py`
(State: export the env vars from .env in your shell; the script reads them via os.getenv.)

Adjust `.env` export syntax — the file uses `KEY=value lines with ` = ` spacing; source it via `bash -c 'set -a; source .env; set +a; .venv/bin/python scripts/agentic_smoke.py'` if needed.

Expected: both profiles return 3–5 products, prices/URLs real (from SerpAPI) or `price unavailable`, disclaimer present, no age/sex/type/race echoed in reasoning text.

- [ ] **Step 3: Debug until fully working**

Fix issues surfaced by the live run (timeouts, malformed JSON, tool parse failing, provider fallbacks). Re-run until clean. Save outputs under `data/{session}/`.

- [ ] **Step 4: Full test suite + commit**

Run: `.venv/bin/python -m pytest tests/agentic -v`
Expected: all pass.

```bash
git add scripts/agentic_smoke.py
git commit -m "test: live agentic smoke script for orchestrate()"
```

---

## Self-Review (performed at plan-writing time)

1. **Spec coverage:** Each spec section maps to tasks: Input contract → `orchestrate(demographics, answers)` (T7); Fitzpatrick mapping → T2 (`FITZPATRICK_MAP`, exhaustive I–VI, primary col); questionnaire → T2; LLM router yml-driven, hybrid, per-provider timeouts → T3; aistack endpoints → T4; research sub-agents → T6; find_products SerpAPI→aistack→OBF + never-invent + budget → T5; orchestrator loop w/ mandatory gate + max turns + thin-result fallback → T7; output contract + disclaimer + error table → T7; testing plan → T1–T8.
2. **Placeholder scan:** No "TBD/TODO" left. The earlier draft's pseudo-`_eval_tool_json`/`build_fallback`-returning-`None` edge cases are resolved by concrete implementations in T7 (extracts, strips, falls back, returns dicts).
3. **Type consistency:** `call_llm` returns `{"type","provider","content","tool_calls"}` (T3) — consumed in `_run_agent_loop` (T7) using `.get`, compatible. `find_products` returns flat dict list — used the same in T5 test assertions. `fitzpatrick_for` returns `(list[int], int)` — T2 tests match. Questionnaire ids `skin_type/primary_concern/...` identical to `session.build_profile` reads. `role="worker"` in research, `role="planner"` in orchestrator where models differ per yml. No type drift.