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