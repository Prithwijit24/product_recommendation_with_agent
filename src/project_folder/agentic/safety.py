"""LLM-based safety agent: evaluates routine + flags, decides if it's safe to proceed."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from . import config, providers
from .models import SafetyVerdict, parse_with_retry


def _safety_cfg() -> dict:
    return config.get_section("safety")


def _fitzpatrick_map() -> dict[str, dict]:
    return _safety_cfg().get("fitzpatrick_map", {})


def fitzpatrick_for(race: str) -> tuple[list[int], int]:
    """Return (band, primary) for a model race label. Race never escapes this module."""
    raw = _fitzpatrick_map()
    entry = raw[race.lower().strip()]
    return entry["types"], entry["primary"]


def _pregnancy_contraindications() -> tuple[str, ...]:
    return tuple(_safety_cfg().get("pregnancy_contraindications", ()))


def _strict_blocked() -> tuple[str, ...]:
    return tuple(_safety_cfg().get("strict_blocked", ()))


def get_safety_prompt() -> str:
    return _safety_cfg().get("prompts", {}).get("system", "")


def _evaluate_safety(ingredients: list[str], flags: dict) -> SafetyVerdict:
    """Call the LLM safety agent and return a validated SafetyVerdict.

    Retries once with error feedback if the response fails validation.
    """
    cfg = _safety_cfg()
    logger.info(f"[safety] evaluating {len(ingredients)} ingredients, flags={flags}")
    user_prompt = (
        f"Ingredients: {json.dumps(ingredients)}\n"
        f"User flags: {json.dumps(flags)}\n"
        "Evaluate safety and respond in strict JSON."
    )
    messages = [
        SystemMessage(content=get_safety_prompt()),
        HumanMessage(content=user_prompt),
    ]
    llm = providers.chat_model(role=cfg.get("llm_role", "worker"))
    max_retries = int(cfg.get("parse_max_retries", 1))
    return parse_with_retry(SafetyVerdict, messages, llm, max_retries=max_retries)


def check_contraindications(
    ingredients: list[str], flags: dict | None = None
) -> tuple[bool, list[str]]:
    """LLM safety gate. flags: dict {pregnancy, sensitivity_fragrance, sensitive_general, ...}.

    Returns (ok, issues). ok=False and non-empty issues if the LLM flags anything.
    """
    flags = flags or {}
    try:
        verdict = _evaluate_safety(ingredients, flags)
        logger.info(f"[safety] verdict: ok={verdict.ok}, issues={verdict.issues}")
        return verdict.ok, verdict.issues
    except Exception as e:  # noqa: BLE001
        logger.error(f"[safety] check_contraindications failed: {e}")
        return True, []


def flagged_ingredients(
    ingredients: list[str], flags: dict | None = None
) -> set[str]:
    """LLM safety gate returning the set of ingredient names that are unsafe.

    Used by the strip-flagged fallback to remove unsafe products.
    """
    flags = flags or {}
    try:
        verdict = _evaluate_safety(ingredients, flags)
        flagged = {f.lower() for f in verdict.flagged_ingredients}
        if flagged:
            logger.info(f"[safety] flagged ingredients: {flagged}")
        return flagged
    except Exception as e:  # noqa: BLE001
        logger.error(f"[safety] flagged_ingredients failed: {e}")
        return set()
