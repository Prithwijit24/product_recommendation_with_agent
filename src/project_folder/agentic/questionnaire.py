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