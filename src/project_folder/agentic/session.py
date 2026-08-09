"""Build session profile + safety flags from demographics and questionnaire answers."""

from __future__ import annotations

from . import config
from .safety import fitzpatrick_for


def _session_cfg() -> dict:
    return config.get_section("session")


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
    cfg = _session_cfg()
    triggers = cfg.get("flag_triggers", {})
    flags = {
        "pregnancy": answers.get("pregnant"),
        "sensitivity_fragrance": triggers.get("fragrance", "") in (answers.get("ensitivities") or []),
        "sensitive_general": answers.get("skin_type") == triggers.get("sensitive_skin", ""),
    }
    return profile_text, flags
