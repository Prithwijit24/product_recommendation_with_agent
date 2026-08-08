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
