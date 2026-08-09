import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from langchain_core.messages import AIMessage, HumanMessage

from project_folder.agentic import research
from project_folder.agentic.models import IngredientSuggestion, SkinConcern


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=5):
        self.calls.append(query)
        return [
            {
                "url": "https://derm.example",
                "title": "Derm review",
                "content": "Melanin density affects hyperpigmentation risk.",
            }
        ]

    def crawl(self, url, only_main_content=True):
        return "# Derm notes\n\nHigh melanin -> more hyperpigmentation."


def fake_llm_json(content_obj):
    def worker(messages):
        assert isinstance(messages[0], HumanMessage) or True
        return AIMessage(content=json.dumps(content_obj))

    return worker


def test_research_skin_concerns_returns_parsed_json():
    out = research.research_skin_concerns(
        "age 25-34, female, fitzpatrick band [1,2]",
        worker=fake_llm_json(
            [
                {
                    "concern": "hyperpigmentation",
                    "evidence": "melanin density increases pigment response",
                    "source": "https://derm.example",
                }
            ]
        ),
        client=FakeClient(),
    )
    assert isinstance(out[0], SkinConcern)
    assert out[0].concern == "hyperpigmentation"


def test_research_ingredients_returns_shortlist():
    out = research.research_ingredients(
        ["hyperpigmentation"],
        worker=fake_llm_json(
            [
                {
                    "ingredient": "niacinamide",
                    "addresses": "hyperpigmentation",
                    "why": "reduces transfer of pigment",
                }
            ]
        ),
        client=FakeClient(),
    )
    assert isinstance(out[0], IngredientSuggestion)
    assert out[0].ingredient == "niacinamide"


def test_research_searches_before_calling_worker():
    client = FakeClient()
    research.research_skin_concerns(
        "profile", worker=fake_llm_json([]), client=client
    )
    assert client.calls  # searched at least once


def test_research_garbage_llm_output_yields_empty():
    worker = lambda messages: AIMessage(content="not json at all")
    out = research.research_ingredients(["acne"], worker=worker, client=FakeClient())
    assert out == []
