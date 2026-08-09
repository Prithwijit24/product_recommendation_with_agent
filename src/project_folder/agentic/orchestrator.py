"""LangGraph orchestrator engine: planner agent + tool sub-agents + safety gate.

Graph topology (compiled per run with a LangChain multi-provider chat router):

    START -> agent -(tool_calls)-> tools -> agent
            agent -(final JSON)-> safety ->(ok)-> finish -> END
            agent -(reject)-> safety ->(retry left)-> agent
            safety  ->(exhausted)-> finish (strip-flagged fallback)

- ``agent``: PlannerRouter (bind_tools) decides the next action.
- ``tools``: LangGraph ToolNode over the session-bound LangChain toolset
  (research_skin_concerns, research_ingredients, find_products, check_contraindications).
- ``safety``: LLM safety agent gate in the graph; a rejection feeds corrective feedback
  straight back into the planner (self-correcting loop, bounded).
- The graph is compiled with a MemorySaver checkpointer keyed by thread_id=session_id,
  and LangSmith tracing is enabled when LANGSMITH_API_KEY is present.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger

from . import config, providers
from .models import RoutineOutput, RoutineProduct, parse_and_validate
from .safety import check_contraindications, flagged_ingredients
from .session import build_profile
from .tools import build_toolset
from .tools import call_tool as react_call_tool


def _orch_cfg() -> dict:
    return config.get_section("orchestrator")


_max_turns_override: int | None = None


def override_max_turns(value: int | None) -> None:
    """Override max_turns for the next run (used by api_runner --max-turns)."""
    global _max_turns_override
    _max_turns_override = value


def _get_max_turns() -> int:
    if _max_turns_override is not None:
        return _max_turns_override
    return int(_orch_cfg().get("max_turns", 14))


def _get_max_safety_retries() -> int:
    return int(_orch_cfg().get("max_safety_retries", 2))


def _get_tool_result_max_chars() -> int:
    return int(_orch_cfg().get("tool_result_max_chars", 2000))


def _get_recursion_limit() -> int:
    return int(_orch_cfg().get("recursion_limit", 80))


def get_system_prompt() -> str:
    prompt = _orch_cfg().get("prompts", {}).get("system", "")
    routine_size = _orch_cfg().get("routine_size", 3)
    # Inject the configured routine size into the prompt
    prompt = prompt.replace("{routine_size}", str(routine_size))
    return prompt


def get_correction_nudge() -> str:
    return _orch_cfg().get("prompts", {}).get("correction_nudge", "")


def get_fallback_routine() -> list[dict]:
    return _orch_cfg().get("fallback_routine", [])


def get_disclaimer() -> str:
    return _orch_cfg().get("disclaimer", "")


# --------------------------------------------------------------------------- #
# parse helpers (kept public for reuse/tests)
# --------------------------------------------------------------------------- #


def extract_routine_json(text: str) -> RoutineOutput | None:
    """Parse and validate a routine JSON response from the planner."""
    try:
        return parse_and_validate(RoutineOutput, text or "")
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to parse routine JSON: %s", e)
        return None


def _extract_tool_json(content: str) -> dict | None:
    try:
        obj = json.loads(content or "")
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from text: %s, Error: %s", content, e)
        return None
    return None


def _build_routine_from_history(msgs: list) -> RoutineOutput:
    """Build a RoutineOutput from tool results when the LLM is stuck in a loop.

    Parses find_products and research_ingredients results to construct a valid routine.
    """

    routine_size = _orch_cfg().get("routine_size", 3)
    products = []
    concerns = []

    for m in msgs:
        if not isinstance(m, ToolMessage):
            continue
        try:
            content = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Extract products from find_products results
        # find_products returns items with "name" + "ingredient" keys
        # research_ingredients returns items with "ingredient" + "addresses" + "why" keys
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "name" in item and "ingredient" in item:
                    products.append(
                        RoutineProduct(
                            product_name=item.get("name", ""),
                            url=item.get("url", ""),
                            price=item.get("price", ""),
                            ingredient=item.get("ingredient", ""),
                            reasoning=item.get("ingredient_match", ""),
                            image_url=item.get("image_url", ""),
                        )
                    )

        # Extract concerns from research_skin_concerns results
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "concern" in item and "evidence" in item:
                    concerns.append(item["concern"])

    # Deduplicate and limit to routine_size
    seen_ingredients = set()
    unique_products = []
    for p in products:
        if p.ingredient.lower() not in seen_ingredients and len(unique_products) < routine_size:
            seen_ingredients.add(p.ingredient.lower())
            unique_products.append(p)

    # Deduplicate concerns
    unique_concerns = list(dict.fromkeys(concerns))

    if not unique_products:
        return build_fallback()

    return RoutineOutput(routine=unique_products, concerns_addressed=unique_concerns)


def safety_gate(routine: RoutineOutput | None, flags: dict) -> tuple[bool, list[str]]:
    if routine is None:
        return True, []
    ingredients = [p.ingredient for p in routine.routine]
    return check_contraindications(ingredients, flags)


def _strip_flagged(routine: RoutineOutput | None, flags: dict) -> RoutineOutput:
    if routine is None:
        return RoutineOutput()
    ingredients = [p.ingredient for p in routine.routine]
    flagged = flagged_ingredients(ingredients, flags)
    kept = [p for p in routine.routine if p.ingredient.lower() not in flagged]
    return RoutineOutput(
        routine=kept,
        concerns_addressed=routine.concerns_addressed,
        disclaimer=routine.disclaimer,
    )


def build_fallback() -> RoutineOutput:
    return RoutineOutput(
        routine=[RoutineProduct(**p) for p in get_fallback_routine()],
    )


# --------------------------------------------------------------------------- #
# LangGraph state machine
# --------------------------------------------------------------------------- #


class OrchestratorState(TypedDict, total=False):
    """Graph state. messages use the langgraph reducer so any node can append."""

    messages: Annotated[list[BaseMessage], add_messages]
    turns: int
    safety_retries: int
    final: RoutineOutput | None
    result: dict


def build_graph(
    planner,
    profile_text: str,
    flags: dict,
    session_id: str,
) -> object:
    """Compile the skincare orchestration StateGraph for this session."""
    tools = build_toolset(profile_text, flags)
    tool_node = ToolNode(tools)

    max_turns = _get_max_turns()
    max_safety_retries = _get_max_safety_retries()
    tool_result_max_chars = _get_tool_result_max_chars()
    correction_nudge = get_correction_nudge()

    def _count_tool_calls(msgs: list, tool_name: str) -> int:
        """Count how many times a tool was called in message history."""
        count = 0
        for m in msgs:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("name") == tool_name:
                        count += 1
        return count

    def _check_contraindications_called(msgs: list) -> bool:
        """Return True if check_contraindications tool result is in history."""
        for m in msgs:
            if isinstance(m, ToolMessage):
                try:
                    content = json.loads(m.content)
                    if isinstance(content, dict) and "ok" in content:
                        return True
                except (json.JSONDecodeError, TypeError):
                    continue
        return False

    def agent_node(state: OrchestratorState) -> dict:
        msgs = list(state.get("messages", []))
        turns = int(state.get("turns", 0))
        if state.get("final") is not None:
            return {}

        # Loop detection: if check_contraindications already ran and the planner
        # is calling tools again, force it to finalize instead.
        if _check_contraindications_called(msgs):
            logger.info("[graph] LOOP DETECTED: check_contraindications already ran, force finalizing")
            return {"turns": turns + 1, "final": _build_routine_from_history(msgs)}

        # Self-correction nudge: last model output was not parseable, not a tool
        # call, and not a tool result waiting for the planner.
        if msgs and isinstance(msgs[-1], AIMessage):
            content = str(msgs[-1].content or "")
            if (
                not msgs[-1].tool_calls
                and not extract_routine_json(content)
                and not _extract_tool_json(content)
            ):
                msgs.append(HumanMessage(content=correction_nudge))

        if turns >= max_turns:
            return {"turns": turns + 1, "final": build_fallback()}

        # ReAct-JSON fallback: execute the tool inline, then ask the planner to
        # continue (degraded providers that rejected native tool calling).
        if msgs and isinstance(msgs[-1], AIMessage):
            tool_json = _extract_tool_json(str(msgs[-1].content or ""))
            if tool_json:
                result = react_call_tool(
                    tool_json.get("tool", ""),
                    tool_json.get("arguments") or {},
                    profile_text,
                    flags,
                    tools,
                )
                rendered = json.dumps(result)[:tool_result_max_chars]
                return {
                    "messages": [
                        HumanMessage(
                            content=f"Tool {tool_json.get('tool')} returned: {rendered}"
                        )
                    ],
                    "turns": turns + 1,
                }

        logger.info(f"[graph] agent turn {turns + 1}/{max_turns}")
        ai = planner.bind_tools(tools).invoke(msgs)
        final: RoutineOutput | None = None
        if ai.tool_calls:
            tc_names = [tc["name"] for tc in ai.tool_calls]
            logger.info(f"[graph] planner called tools: {tc_names}")
        else:
            final = extract_routine_json(str(ai.content or ""))
            if final:
                logger.info(f"[graph] planner produced final JSON with {len(final.routine)} products")
            else:
                logger.warning(f"[graph] planner produced no valid JSON: {str(ai.content)[:200]}")
        return {"messages": [ai], "turns": turns + 1, "final": final}

    def safety_node(state: OrchestratorState) -> dict:
        final = state.get("final")
        ok, issues = safety_gate(final, flags)
        if ok:
            logger.info("[graph] safety gate PASSED")
            return {}
        logger.warning(f"[graph] safety gate REJECTED: {issues}")
        retries = int(state.get("safety_retries", 0))
        if retries + 1 > max_safety_retries:
            logger.warning("[graph] max safety retries exhausted, using fallback")
            return {}
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "Safety gate rejected the routine with these issues: "
                        f"{issues}. Remove or replace the flagged ingredients. "
                        f"Attempt {retries + 1} of {max_safety_retries}."
                    )
                )
            ],
            "safety_retries": retries + 1,
        }

    def finish_node(state: OrchestratorState) -> dict:
        final = state.get("final") or build_fallback()
        if final is None:
            final = build_fallback()
            logger.warning("[graph] no final routine, using fallback")
        stripped = _strip_flagged(final, flags)
        if not stripped.routine:
            stripped = build_fallback()
            logger.warning("[graph] all products stripped by safety, using fallback")
        logger.info(f"[graph] finish: {len(stripped.routine)} products, concerns={stripped.concerns_addressed}")
        return {
            "result": {
                "session_id": session_id,
                "routine": [p.model_dump() for p in stripped.routine],
                "concerns_addressed": stripped.concerns_addressed,
                "disclaimer": get_disclaimer(),
            }
        }

    def after_agent(state: OrchestratorState) -> str:
        """Agent turn done: route by the nature of the last message."""
        if state.get("final") is not None:
            return "safety"
        turns = int(state.get("turns", 0))
        if turns > max_turns:
            return "finish"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        # Final-answer JSON flagged in state -> safety; anything else (ReAct-JSON
        # tool results, nudges, plain text) loops back to the planner.
        return "agent"

    def after_safety(state: OrchestratorState) -> str:
        """Safety verdict route: acceptable -> finish, otherwise loop with feedback."""
        final = state.get("final")
        ok, _ = safety_gate(final, flags)
        if ok:
            return "finish"
        retries = int(state.get("safety_retries", 0))
        return "agent" if retries < max_safety_retries else "finish"

    graph = StateGraph(OrchestratorState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("safety", safety_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {
            "tools": "tools",
            "safety": "safety",
            "agent": "agent",
            "finish": "finish",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "safety", after_safety, {"agent": "agent", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# Public entry point (unchanged contract for the UI)
# --------------------------------------------------------------------------- #


def orchestrate(
    demographics: dict, answers: dict, session_id: str | None = None
) -> dict:
    import time

    session_id = session_id or str(uuid.uuid4())
    t0 = time.monotonic()
    logger.info(f"[orchestrate] START session={session_id}")
    logger.info(f"[orchestrate] demographics={json.dumps(demographics)}")
    logger.info(f"[orchestrate] answers={json.dumps(answers)}")

    try:
        providers.configure_langsmith()
        profile_text, flags = build_profile(demographics, answers)
        logger.info(f"[orchestrate] profile={profile_text}")
        logger.info(f"[orchestrate] flags={flags}")

        planner = providers.chat_model(role="planner")
        planner_name = getattr(planner, "model_name", str(planner))
        logger.info(f"[orchestrate] planner model={planner_name}")

        graph = build_graph(planner, profile_text, flags, session_id=session_id)
        user_req = (
            f"Profile: {profile_text}\n"
            f"Questionnaire answers: {json.dumps(answers)}\n"
            "Now run the research pipeline and return the routine JSON."
        )
        logger.info("[orchestrate] invoking graph...")
        out = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=get_system_prompt()),
                    HumanMessage(content=user_req),
                ]
            },
            config={"recursion_limit": _get_recursion_limit(), "configurable": {"thread_id": session_id}},
        )
        result = out["result"]
        elapsed = time.monotonic() - t0
        logger.info(f"[orchestrate] DONE in {elapsed:.1f}s, {len(result.get('routine', []))} products")
        return result
    except Exception as e:  # noqa: BLE001 - last-resort fallback
        elapsed = time.monotonic() - t0
        logger.error(f"[orchestrate] FAILED after {elapsed:.1f}s: {e}")
        fb = build_fallback()
        return {
            "session_id": session_id,
            "routine": [p.model_dump() for p in fb.routine],
            "concerns_addressed": [],
            "disclaimer": get_disclaimer(),
            "error": str(e),
        }
