import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import research


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
        return "# Derm notes\n\nHigh melanin → more hyperpigmentation."


def test_research_skin_concerns_returns_parsed_json(monkeypatch):
    def fake_llm(messages, tools=None, *, role="planner"):
        assert role == "worker"
        return {
            "type": "text",
            "content": json.dumps(
                [
                    {
                        "concern": "hyperpigmentation",
                        "evidence": "melanin density increases pigment response",
                        "source": "https://derm.example",
                    }
                ]
            ),
            "tool_calls": [],
            "provider": "agnes",
        }

    client = FakeClient()
    out = research.research_skin_concerns(
        "age 25-34, female, fitzpatrick band [1,2]", worker=fake_llm, client=client
    )
    assert out[0]["concern"] == "hyperpigmentation"
    assert client.calls  # searched at least once


def test_research_ingredients_returns_shortlist(monkeypatch):
    def fake_llm(messages, tools=None, *, role="planner"):
        return {
            "type": "text",
            "content": json.dumps(
                [
                    {
                        "ingredient": "niacinamide",
                        "addresses": "hyperpigmentation",
                        "why": "reduces transfer of pigment",
                    }
                ]
            ),
            "tool_calls": [],
            "provider": "agnes",
        }

    client = FakeClient()
    out = research.research_ingredients(["hyperpigmentation"], worker=fake_llm, client=client)
    assert out[0]["ingredient"] == "niacinamide"
