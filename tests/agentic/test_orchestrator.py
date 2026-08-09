import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from langchain_core.messages import AIMessage

from project_folder.agentic import orchestrator
from project_folder.agentic.models import (
    RoutineOutput,
    RoutineProduct,
    SafetyVerdict,
    SkinConcern,
)


def test_extract_routine_json_from_fenced_block():
    text = '```json\n{"routine": [], "concerns_addressed": [], "disclaimer": "x"}\n```'
    out = orchestrator.extract_routine_json(text)
    assert isinstance(out, RoutineOutput)
    assert out.routine == []


def test_extract_routine_json_returns_none_on_invalid():
    assert orchestrator.extract_routine_json("not json") is None


def test_safety_gate_blocks_retinol_on_pregnancy():
    routine = RoutineOutput(
        routine=[RoutineProduct(product_name="A", ingredient="retinol")],
    )
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["Not recommended during pregnancy: retinol"],
            flagged_ingredients=["retinol"],
        ),
    ):
        ok, issues = orchestrator.safety_gate(routine, {"pregnancy": "yes"})
    assert not ok
    assert any("retino" in i.lower() for i in issues)


class ScriptedPlanner:
    """LangChain model double: pipelines pre-built AIMessages, records calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self.script[min(self.calls - 1, len(self.script) - 1)]


GOOD_JSON = (
    '{"routine": [{"product_name": "Niacinamide serum", "url": "https://x", "price": "$9", '
    '"ingredient": "niacinamide", "reasoning": "r"}], "concerns_addressed": ["tone"], '
    '"disclaimer": "d"}'
)

STUBBORN_JSON = (
    '{"routine": [{"product_name": "Tretinoin serum", "url": "", "price": "", '
    '"ingredient": "tretinoin", "reasoning": "r"}], "concerns_addressed": ["acne"], '
    '"disclaimer": "d"}'
)


def test_orchestrate_fallback_when_llm_dead(monkeypatch):
    monkeypatch.setattr(
        orchestrator.providers, "chat_model", lambda role="planner": (_ for _ in ()).throw(RuntimeError("all providers down"))
    )
    out = orchestrator.orchestrate(
        {"age_range": "25-34", "sex": {"value": "F"}, "race": {"value": "Asian"}},
        {"skin_type": "combination", "pregnancy": "no", "budget": "medium"},
    )
    assert out["routine"]
    assert "disclaimer" in out
    assert len(out["routine"]) >= 3
    assert "error" in out


def _fake_skin_concerns(profile, worker=None, client=None):
    return [SkinConcern(concern="tone", evidence="e", source="u")]


def _fake_ingredients(concerns, worker=None, client=None):
    return []


def _fake_products(ing, budget="medium"):
    return []


def test_orchestrate_executes_tools_then_finalizes():
    planner = ScriptedPlanner(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "research_skin_concerns",
                        "args": {"profile": "p"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=GOOD_JSON),
        ]
    )
    with patch(
        "project_folder.agentic.orchestrator.providers.chat_model", lambda role: planner
    ), patch(
        "project_folder.agentic.research.research_skin_concerns", _fake_skin_concerns
    ), patch(
        "project_folder.agentic.research.research_ingredients", _fake_ingredients
    ), patch(
        "project_folder.agentic.products.find_products", _fake_products
    ), patch(
        "project_folder.agentic.safety._evaluate_safety",
        lambda ingredients, flags: SafetyVerdict(ok=True),
    ):
        out = orchestrator.orchestrate(
            {"age_range": "25-34", "sex": {"value": "F"}, "race": {"value": "Indian"}},
            {"skin_type": "combination", "pregnancy": "no", "budget": "medium"},
        )
    assert planner.calls == 2  # tool round-trip then final JSON
    assert len(out["routine"]) == 1
    assert out["routine"][0]["product_name"] == "Niacinamide serum"
    assert out["routine"][0]["price"] == "$9"


def test_orchestrate_stubborn_violation_ends_in_fallback(monkeypatch):
    planner = ScriptedPlanner([AIMessage(content=STUBBORN_JSON)])
    monkeypatch.setattr(orchestrator.providers, "chat_model", lambda role: planner)
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["Not recommended during pregnancy: tretinoin"],
            flagged_ingredients=["tretinoin"],
        ),
    ):
        out = orchestrator.orchestrate(
            {"age_range": "25-34", "sex": {"value": "F"}, "race": {"value": "Asian"}},
            {"skin_type": "combination", "pregnant": "yes", "budget": "medium"},
        )
    assert out["routine"]
    assert [p["product_name"] for p in out["routine"]] == [
        p["product_name"] for p in orchestrator.get_fallback_routine()
    ]
    assert "disclaimer" in out


def test_strip_flagged_keeps_products_for_sensitive_user():
    routine = RoutineOutput(
        routine=[RoutineProduct(product_name="X", ingredient="niacinamide")],
    )
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(ok=True),
    ):
        out = orchestrator._strip_flagged(routine, {"sensitive_general": True})
    assert len(out.routine) == 1


def test_strip_flagged_removes_unsafe_ingredients():
    routine = RoutineOutput(
        routine=[
            RoutineProduct(product_name="A", ingredient="retinol"),
            RoutineProduct(product_name="B", ingredient="niacinamide"),
        ],
    )
    with patch(
        "project_folder.agentic.safety._evaluate_safety",
        return_value=SafetyVerdict(
            ok=False,
            issues=["pregnancy: retinol"],
            flagged_ingredients=["retinol"],
        ),
    ):
        out = orchestrator._strip_flagged(routine, {"pregnancy": "yes"})
    assert len(out.routine) == 1
    assert out.routine[0].ingredient == "niacinamide"
