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


def test_strip_flagged_keeps_products_for_sensitive_user():
    routine = {"routine": [{"product_name": "X", "ingredient": "niacinamide"}]}
    out = orchestrator._strip_flagged(routine, {"sensitive_general": True})
    assert len(out["routine"]) == 1


def test_enforce_routine_keys_renames_name():
    out = orchestrator._enforce_routine_keys(
        [{"name": "CeraVe", "url": "", "price": "", "ingredient": "niacinamide"}]
    )
    assert out[0]["product_name"] == "CeraVe"
    assert "name" not in out[0]
    out2 = orchestrator._enforce_routine_keys([{"product_name": "Already"}])
    assert out2[0] == {"product_name": "Already", "price": "price unavailable"}


def test_enforce_routine_keys_maps_empty_price():
    out = orchestrator._enforce_routine_keys(
        [{"product_name": "X", "price": "", "url": "u", "ingredient": "i", "reasoning": "r"}]
    )
    assert out[0]["price"] == "price unavailable"
    out2 = orchestrator._enforce_routine_keys([{"product_name": "Y"}])
    assert out2[0]["price"] == "price unavailable"
    out3 = orchestrator._enforce_routine_keys([{"product_name": "Z", "price": "12.99"}])
    assert out3[0]["price"] == "12.99"


def test_stripped_empty_uses_fallback(monkeypatch):
    stubborn_json = (
        '```json\n{"routine": [{"product_name": "Tretinoin serum", "url": "", "price": "", '
        '"ingredient": "tretinoin", "reasoning": "r"}], "concerns_addressed": ["acne"], '
        '"disclaimer": "d"}\n```'
    )

    def stubborn_llm(*a, **k):
        return {"type": "content", "content": stubborn_json}

    monkeypatch.setattr(orchestrator.providers, "call_llm", stubborn_llm)
    out = orchestrator.orchestrate(
        {"age_range": "25-34", "sex": {"value": "F"}, "race": {"value": "Asian"}},
        {"skin_type": "combination", "pregnant": "yes", "budget": "medium"},
    )
    assert out["routine"]
    assert [p["product_name"] for p in out["routine"]] == [
        p["product_name"] for p in orchestrator.FALLBACK_ROUTINE
    ]
    assert "disclaimer" in out