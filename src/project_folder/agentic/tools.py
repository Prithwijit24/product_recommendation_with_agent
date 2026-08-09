"""LangChain tools the planner agent can call: research, products, safety gate.

Tools are built per-session (:func:`build_toolset`) so the safety flags
and the demographic profile text can be bound at graph-build time.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from . import config, products, research
from .safety import check_contraindications


def _tools_cfg() -> dict:
    return config.get_section("tools")


def build_toolset(profile_text: str, flags: dict) -> list[BaseTool]:
    """Session-bound LangChain tools for one orchestrate() run."""
    cfg = _tools_cfg()
    descriptions = cfg.get("descriptions", {})

    @tool("research_skin_concerns", description=descriptions.get("research_skin_concerns", ""))
    def research_skin_concerns(profile: str = "") -> str:
        """Infer likely skin-science considerations for the user's profile."""
        items = research.research_skin_concerns(profile or profile_text)
        return json.dumps([i.model_dump() for i in items])

    @tool("research_ingredients", description=descriptions.get("research_ingredients", ""))
    def research_ingredients(concerns: list[str]) -> str:
        """Shortlist cosmetic ingredients for the given concerns."""
        items = research.research_ingredients(concerns)
        return json.dumps([i.model_dump() for i in items])

    @tool("find_products", description=descriptions.get("find_products", ""))
    def find_products(ingredients: list[str], budget: str = "medium") -> str:
        """Find one real purchasable product per ingredient."""
        items = products.find_products(ingredients, budget=budget)
        return json.dumps(items)

    @tool("check_contraindications", description=descriptions.get("check_contraindications", ""))
    def check_contraindications_tool(ingredients: list[str]) -> str:
        """Safety gate over the candidate ingredient list for this profile."""
        ok, issues = check_contraindications(ingredients, flags)
        return json.dumps({"ok": ok, "issues": issues})

    return [
        research_skin_concerns,
        research_ingredients,
        find_products,
        check_contraindications_tool,
    ]


def call_tool(name: str, args: dict, profile_text: str, flags: dict, toolset: list[BaseTool]):
    """ReAct-JSON fallback dispatcher: executes a tool by name with plain dict args.

    Used when a degraded provider has no native tool calling; mirrors exactly what
    the LangGraph ToolNode executes for the native path.
    """
    from langchain_core.tools import ToolException

    for t in toolset:
        if t.name == name:
            try:
                return t.invoke(args)
            except ToolException as exc:
                return {"error": str(exc)}
    return {"error": f"unknown tool: {name}"}
