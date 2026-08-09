import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic.models import SafetyVerdict
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
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["Not recommended during pregnancy: retinol serum"],
            flagged_ingredients=["retinol serum"],
        ),
    ):
        ok, _ = check_contraindications(["retinol serum"], {"pregnancy": "yes"})
    assert not ok


def test_pregnancy_salicylic_flagged():
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["Not recommended during pregnancy: salicylic acid toner"],
            flagged_ingredients=["salicylic acid toner"],
        ),
    ):
        ok, _ = check_contraindications(["salicylic acid toner"], {"pregnancy": "yes"})
    assert not ok


def test_fragrance_sensitive_flagged():
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["Fragrance-sensitive skin: avoid rosewater fragrance"],
            flagged_ingredients=["rosewater fragrance"],
        ),
    ):
        ok, _ = check_contraindications(["rosewater fragrance"], {"sensitivity_fragrance": True})
    assert not ok


def test_clean_ingredients_pass():
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(ok=True),
    ):
        ok, issues = check_contraindications(
            ["niacinamide", "hyaluronic acid"], {"pregnancy": "no"}
        )
    assert ok
    assert issues == []


def test_sensitive_general_advisory_does_not_block():
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=True,
            issues=["Sensitive skin — patch test any new product before full use."],
        ),
    ):
        ok, issues = check_contraindications(["niacinamide"], {"sensitive_general": True})
    assert ok
    assert any("patch test" in i.lower() for i in issues)


def test_sensitive_general_with_blocked_ingredient():
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["Not recommended during pregnancy: retinol"],
            flagged_ingredients=["retinol"],
        ),
    ):
        ok, issues = check_contraindications(
            ["retinol"], {"sensitive_general": True, "pregnancy": "yes"}
        )
    assert not ok
    assert any("pregnancy" in i.lower() for i in issues)


def test_llm_failure_fails_open():
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        side_effect=RuntimeError("provider down"),
    ):
        ok, issues = check_contraindications(["retinol"], {"pregnancy": "yes"})
    assert ok
    assert issues == []
