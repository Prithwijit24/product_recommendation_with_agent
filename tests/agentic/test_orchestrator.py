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