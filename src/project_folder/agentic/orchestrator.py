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
        ok, _ = check_contraindications([p.get("ingredient", "")], flags)
        if ok:
            kept.append(p)
    return {**routine_json, "routine": kept}


def _enforce_routine_keys(routine: list[dict]) -> list[dict]:
    out = []
    for item in routine:
        item = dict(item)
        if "product_name" not in item and "name" in item:
            item["product_name"] = item.pop("name")
        out.append(item)
    return out


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
            "routine": _enforce_routine_keys(result.get("routine", [])),
            "concerns_addressed": result.get("concerns_addressed", []),
            "disclaimer": DISCLAIMER,
        }
    except Exception as e:  # noqa: BLE001 - last-resort fallback
        fb = build_fallback()
        return {
            "session_id": session_id,
            "routine": _enforce_routine_keys(fb["routine"]),
            "concerns_addressed": [],
            "disclaimer": DISCLAIMER,
            "error": str(e),
        }