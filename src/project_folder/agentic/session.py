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
