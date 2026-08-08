# AI Skincare Recommendation — Agentic Orchestrator Design

Date: 2026-08-08 · Status: Approved for implementation

## 1. Big Picture

A face image is processed by the existing ML pipeline (age / gender / race models via
`src/project_folder/main.py`). The agentic orchestrator consumes only the resulting demographic
profile `{age_range, sex, race}` plus answers to a fixed skin questionnaire and produces a
personalized cosmetic skincare routine by actively researching live sources. No second ML model.

```
Face image → ML models {age, sex, race} → [Fitzpatrick mapping] + questionnaire answers
        → Orchestrator Agent (LLM)
            → research_skin_concerns (sub-agent, aistack search/crawl, derm-scoped)
            → research_ingredients (sub-agent) → ingredient shortlist
            → find_products (SerpAPI primary → aistack /search → OBF fallback)
            → check_contraindications (deterministic, MANDATORY gate)
        → Final routine JSON {products, concerns_addressed, disclaimer}
```

## 2. Input Contract

```json
{
  "session_id": "string",
  "timestamp": "ISO8601",
  "demographics": {
    "age_range": "25-34",
    "sex": {"value": "F", "confidence": 0.87},
    "race": {"value": "Asian", "confidence": 0.74}
  },
  "answers": {
    "skin_type": "oily|dry|combination|normal|sensitive",
    "primary_concern": "acne|pigmentation|aging|redness|dullness",
    "sun_exposure": "low|moderate|high",
    "sensitivities": ["none"] | ["fragrance"] | ["retinoids"] | ["aha_bha"],
    "budget": "high|medium|low",
    "pregnant": "yes|no|prefer_not_to_say"
  }
}
```

`answers` come from the fixed pure-Python questionnaire (Section 4). The orchestrator is only
called after both demographics and questionnaire answers exist.

## 3. Fitzpatrick Mapping (exhaustive, fixed table)

The ML model predicts race into 4 categories: `asian`, `indian`, `white`, `black`.
Race itself is NEVER used for product selection or in user-facing output. It is converted
to a Fitzpatrick band + primary type via this fixed mapping:

| Race (model)   | Types (band, exhaustive union = {1..7}) | Primary |
|----------------|-----------------------------------------|---------|
| `white`        | [1]                                     | 1       |
| `asian`        | [1, 2]                                  | 1       |
| `indian`       | [2, 3]                                  | 2       |
| `black`        | [4, 5, 6, 7]                            | 4       |

Primary = most conservative type in the band (assume the most sun-sensitive end; safest
default for ingredient/sunscreen matching). Adjustable as a single config value per row.

Research prompts ground reasoning in Fitzpatrick biology (melanin density, sun-sensitivity,
photoaging pattern), never race. Agent output must never reference race, age, sex.

## 4. Questionnaire (fixed, pure Python, no LLM)

Static list of 5 questions (mirrors the retail PC/Budget/Location select pattern):

1. skin_type — Oily / Dry / Combination / Normal / Sensitive
2. primary_concern — Acne & breakouts / Uneven tone & pigmentation / Fine lines & aging / Redness & sensitivity / Dullness
3. sun_exposure — Mostly indoor / Moderate outdoor / Significant outdoor
4. sensitivities_actives — None / Fragrance-sensitive / Retinoid-dependent / AHA-BHA active
5. budget — High / Medium / Low
(+ pregnancy flag captured in profile: yes/no/prefer-not-to-say → passes into safety flags)

Rendered mechanically as selectboxes by callers (e.g. Streamlit later); the module returns the
static question list via `get_questionnaire()` so UI can't drift from backend.

## 5. LLM Router (pure Python, `api_config.yml`-driven)

Providers and their order come ONLY from `src/project_folder/config/api_config.yml`
(agnes → opencode → llm7io → oraclellm). Env vars hold keys (`AGNES_API_KEY`,
`OPENCODE_API_KEY`, `LLM7IO_API_KEY`, `ORACLELLM_API_KEY`).

- Priority sequence is the yml file order.
- A provider is "available" iff its key is set in env.
- `planner_model` = orchestrator LLM; `worker_model` = sub-agent LLM (both from yml).
- Per-provider timeouts from yml (default 60s; oraclellm planner 300s / worker 240s).
- Call model: **hybrid (C)** — attempt native OpenAI tool-calling first; on tool-unsupported
  error or timeout, degrade that provider to ReAct JSON mode (agent replies
  `{"tool":..., "args":...}` as text) for the remainder of the session.
- On live failure (rate limit, timeout, malformed), fall through to next provider in order.
- Any provider that hard-fails is marked degraded for the session.

All providers are openai-compatible; implement via `openai` SDK (pinned in requirements)
with per-provider `base_url`/`api_key`/`model`, or raw httpx where trivial.

## 6. Tools & Sub-Agents

| Tool | Type | Behavior | Backing |
|------|------|----------|---------|
| `research_skin_concerns` | sub-agent | 1–3 search/crawl rounds, derm-scoped sources; returns condensed concerns + one-line evidence each. Grounds in Fitzpatrick/age/hormonal biology, not race. | aistack `/search` + `/crawl` (domain-scoped query strings) |
| `research_ingredients` | sub-agent | searches ingredients per concern; returns shortlist + why | same aistack endpoints |
| `find_products` | tool (single call) | ingredient → real products `{name, url, price, ingredient_match}` | **SerpAPI google_shopping primary** (env `SERP_API_KEY`) → aistack `/search` retail-scoped → Open Beauty Facts API (last resort). Never invents products/prices/URLs. |
| `check_contraindications` | deterministic function, NO LLM | validates ingredient list against static unsafe-combination table + user flags (pregnancy, sensitivities) | pure Python table in `safety.py` |
| `log_interaction` | tool | records session_id → routine (anonymized) | in-memory/store: optional, minimal (deferred per build order discussion) |

`find_products` specifics:
- SerpAPI `google_shopping` engine: query per ingredient, parse organic results
  `{title, price, link, source}`. e-commerce sources only.
- Clip/limit to budget tier (price filter if engine supports, else pick reasonable).
- `ingredient_match`: which shortlisted ingredient the product addresses (LLM-lite: deterministic
  keyword match product title vs ingredient; fallback = best title token match).

## 7. Orchestrator Loop (system prompt + hard safety gate)

System prompt follows the spec Section 3 verbatim (research concerns → ingredients → products →
MANDATORY check_contraindications → finalize; cosmetic framing only; never echo demographics;
thin/low-confidence results → best-seller fallback; medical-condition signals → dermatologist
referral). Response must be strictly-valid JSON matching the output contract.

Hard gate (code, not prompt): orchestrator loop runs max 6 tool rounds; after the agent emits
its final routine, backend re-runs `check_contraindications` on every ingredient in the routine:

```python
def safety_gate(routine_json) -> (ok: bool, flags: list[str]):
    ingredients = {p["ingredient"] for p in routine_json["routine"]}
    return check_contraindications(ingredients, profile_flags)
```

- `ok=False` → feed flags back to the agent and retry (max 2 retries) → then strip/replace
  the offending products deterministically and emit routine + disclaimer.
- No routine is returned unless the final ingredient list passes the gate.

Max agent turns: 6. Sub-agents get their own small loops (max 3 rounds each).

Fallback branch: if ANY tool result is thin/conflicting (empty search reply, zero products, all
providers down), return general best-seller routine (hardcoded kosher set) with disclaimer —
never a hallucinated product.

## 8. Output Contract

```json
{
  "session_id": "string",
  "routine": [
    {
      "product_name": "string",
      "url": "string",
      "price": "string",
      "ingredient": "string",
      "reasoning": "one plain-language sentence"
    }
  ],
  "concerns_addressed": ["string"],
  "disclaimer": "Cosmetic recommendations only, not a medical diagnosis. Consult a dermatologist for persistent or worsening skin issues."
}
```

- Backend renders Buy Now from `url`; FTC disclosure next to affiliate links (future).
- Disclaimer present in every response.

## 9. Error Handling

| Failure | Behavior |
|---------|----------|
| provider fails | fall to next in yml order; mark degraded |
| provider unsupported tools | degrade that provider to ReAct JSON |
| research tool fails | retry once → best-seller fallback branch |
| product tool fails | SerpAPI → search → OBF cascade; all fail → routine without prices marked "price unavailable" |
| safety gate flags | feed back to agent, retry ≤2, then deterministic strip/replace |
| session crash anywhere | graceful fallback JSON + disclaimer; log error |

## 10. Testing Plan (run, debug, resolve)

1. `test_safety.py` — deterministic contraindication unit tests (pregnancy+tretinoin, AHA+BHA,
   retinol+masked ingredient, etc.).
2. `test_providers.py` — priority/availability from yml order; fallback behavior with env toggles.
3. Sub-agent smoke: each independently queried with fixed profiles
   (25-34 F asian, 55-64 M white, 18-24 F black, 35-44 F indian) against live aistack.
4. `find_products` smoke with 3 known ingredients (niacinamide, HA, retinoid) → verify real
   price/link shape from SerpAPI.
5. End-to-end `orchestrate()` write `data/{session_id}/routine.json`; inspect for contract
   validity, no invented URLs, safety gate passed, no demographic echo.
6. Run batch of sample outputs for dermatologist review (user's build-order item 8).

## 11. Out of Scope (this iteration)

- Streamlit UI integration (later; module exposes `orchestrate()` + `get_questionnaire()`).
- Image/ML pipeline changes, race→Fitzpatrick swap already covered (fixed table).
- Analytics logging UI; `log_interaction` minimal stub only.
- Anthropic/OpenAI/Groq provider atples from README spec (not present in user's env).