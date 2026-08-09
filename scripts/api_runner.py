"""api_runner.py: live run/debug/test harness for the agentic skincare flow.

Funnels the whole pipeline (demographics -> research -> products -> safety gate)
into one CLI so you can iterate fast:

    python scripts/api_runner.py                      # default (app-f preset)
    python scripts/api_runner.py --profile app-m      # another preset
    python scripts/api_runner.py --race black --age 55-64 --sex M \
        --build high --sun "Significant outdoor"
    python scripts/api_runner.py --debug              # stream every graph node update
    python scripts/api_runner.py --expect-products 3 --out run.json

Exit codes: 0 = ok, 1 = exception / expectation failed, 2 = routine empty.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from langchain_core.messages import HumanMessage, SystemMessage

from project_folder.agentic import orchestrator as orch
from project_folder.agentic import providers
from project_folder.agentic.session import build_profile

PRESETS = {
    "app-f": {
        "demographics": {
            "age_range": "25-34",
            "sex": {"value": "F", "confidence": 0.87},
            "race": {"value": "Asian", "confidence": 0.74},
        },
        "answers": {
            "skin_type": "Combination",
            "primary_concern": "Uneven tone & pigmentation",
            "sun_exposure": "Moderate outdoor",
            "sensitivities": ["None"],
            "budget": "medium",
            "pregnant": "no",
        },
    },
    "app-m": {
        "demographics": {
            "age_range": "55-64",
            "sex": {"value": "M", "confidence": 0.9},
            "race": {"value": "Black", "confidence": 0.8},
        },
        "answers": {
            "skin_type": "Oily",
            "primary_concern": "Fine lines & aging",
            "sun_exposure": "Significant outdoor",
            "sensitivities": ["None"],
            "budget": "high",
            "pregnant": "no",
        },
    },
}

RACES = ("white", "asian", "indian", "black")
SEXES = ("M", "F")
SKIN_TYPES = ("Oily", "Dry", "Combination", "Normal", "Sensitive")
CONCERNS = (
    "Acne & breakouts",
    "Uneven tone & pigmentation",
    "Fine lines & aging",
    "Redness & sensitivity",
    "Dullness",
)
SUN_EXPOSURES = ("Mostly indoor", "Moderate outdoor", "Significant outdoor")
SENSITIVITIES = ("None", "Fragrance-sensitive", "Retinoid-dependent", "AHA-BHA active")
BUDGETS = ("low", "medium", "high")


def pick_enum(name: str, value: str, allowed: tuple[str, ...]) -> str:
    for candidate in allowed:
        if value.lower() in candidate.lower():
            return candidate
    raise SystemExit(f"bad {name}: {value!r} (choose from {', '.join(allowed)})")


def collect_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.preset != "custom":
        name = args.preset
        data = PRESETS[name]
        return dict(data["demographics"]), dict(data["answers"])
    demographics = {
        "age_range": args.age_range,
        "sex": {"value": pick_enum("sex", args.sex, SEXES)},
        "race": {"value": pick_enum("race", args.race, RACES)},
    }
    answers = {
        "skin_type": pick_enum("skin_type", args.skin_type, SKIN_TYPES),
        "primary_concern": pick_enum("primary_concern", args.concern, CONCERNS),
        "sun_exposure": pick_enum("sun_exposure", args.sun, SUN_EXPOSURES),
        "sensitivities": [
            pick_enum("sensitivities", s.strip(), SENSITIVITIES) for s in args.sensitivities.split(",")
        ],
        "budget": pick_enum("budget", args.budget, BUDGETS),
        "pregnant": args.pregnant,
    }
    return demographics, answers


def summarize_update(node: str, update: dict) -> str:
    msgs = update.get("messages")
    if msgs:
        last = msgs[-1]
        kind = type(last).__name__
        if kind == "ToolMessage":
            return f"{node:8} tool result: {(str(last.content) or '')[:140]}"
        if kind == "AIMessage":
            if getattr(last, "tool_calls", None):
                calls = ", ".join(tc["name"] for tc in last.tool_calls)
                return f"{node:8} tool calls: {calls}"
            return f"{node:8} text: {(str(last.content) or '')[:140]}"
        if kind == "HumanMessage":
            return f"{node:8} human input: {(str(last.content) or '')[:140]}"
    return f"{node:8} state update: {list(update.keys())}"


def run_debug(demographics: dict, answers: dict, session_id: str) -> dict:
    """Stream the compiled LangGraph step by step; return the same dict orchestrate() would."""
    providers.configure_langsmith()
    profile_text, flags = build_profile(demographics, answers)
    planner = providers.chat_model(role="planner")
    graph = orch.build_graph(planner, profile_text, flags, session_id=session_id)
    user_req = (
        f"Profile: {profile_text}\n"
        f"Questionnaire answers: {json.dumps(answers)}\n"
        "Now run the research pipeline and return the routine JSON."
    )
    from project_folder.agentic.config import get_section

    orch_cfg = get_section("orchestrator")
    recursion_limit = int(orch_cfg.get("recursion_limit", 80))
    config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": session_id}}
    messages = [
        SystemMessage(content=orch.get_system_prompt()),
        HumanMessage(content=user_req),
    ]
    try:
        for chunk in graph.stream(
            {"messages": messages}, config=config, stream_mode="updates"
        ):
            if not chunk:
                continue
            for node, update in chunk.items():
                if update is not None:
                    print(summarize_update(node, update))
        state = graph.get_state(config)
        result = state.values.get("result")
        if result is None:
            raise RuntimeError("graph finished without a result (planner exhausted turns?)")
    except Exception as exc:  # noqa: BLE001 - report, do not crash the whole script
        fallback = orch.build_fallback()
        return {
            "session_id": session_id,
            "routine": [p.model_dump() for p in fallback.routine],
            "concerns_addressed": [],
            "disclaimer": orch.get_disclaimer(),
            "error": str(exc),
        }
    return result


def pprint_routine(result: dict) -> None:
    print("=" * 78)
    print(f"session_id : {result.get('session_id')}")
    print(f"concerns   : {result.get('concerns_addressed') or []}")
    if result.get("error"):
        print(f"error      : {result['error']}")
    routine = result.get("routine") or []
    for i, item in enumerate(routine, 1):
        print("-" * 78)
        print(f"{i}. {item.get('product_name', '')}")
        print(f"     price      : {item.get('price', '')}")
        print(f"     url        : {item.get('url', '')}")
        print(f"     image      : {item.get('image_url', '')}")
        print(f"     ingredient : {item.get('ingredient', '')}")
        print(f"     why        : {item.get('reasoning', '')}")
    print("-" * 78)
    print(f"disclaimer : {result.get('disclaimer', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="app-f", choices=["app-f", "app-m", "custom"],
                        help="prebuilt profile (implied demographics/answers); 'custom' uses the flags below")
    parser.add_argument("--age-range", default="25-34")
    parser.add_argument("--sex", default="F", choices=SEXES)
    parser.add_argument("--race", default="asian", choices=RACES)
    parser.add_argument("--skin-type", default="Combination")
    parser.add_argument("--concern", default="Uneven tone & pigmentation")
    parser.add_argument("--sun", default="Moderate outdoor")
    parser.add_argument("--sensitivities", default="None")
    parser.add_argument("--budget", default="medium", choices=BUDGETS)
    parser.add_argument("--pregnant", default="no", choices=("yes", "no"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--max-turns", type=int, default=None, help="override orchestrator MAX_TURNS")
    parser.add_argument("--debug", action="store_true", help="stream LangGraph node updates")
    parser.add_argument("--expect", type=int, default=0, metavar="N",
                        help="fail (exit 1) unless routine has >= N products")
    parser.add_argument("--out", type=Path, default=None, help="write full JSON result here")
    args = parser.parse_args()

    if args.max_turns is not None:
        orch.override_max_turns(args.max_turns)

    demographics, answers = collect_inputs(args)
    session_id = args.session_id or f"runner-{uuid.uuid4().hex[:8]}"
    print(f"[input] demographics : {json.dumps(demographics)}")
    print(f"[input] answers      : {json.dumps(answers)}")
    print(f"[input] session_id   : {session_id}")

    t0 = time.monotonic()
    try:
        if args.debug:
            result = run_debug(demographics, answers, session_id)
        else:
            result = orch.orchestrate(demographics, answers, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 - the harness must survive backend failures
        print(f"[runner] exception: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - t0
    print(f"[runner] finished in {elapsed:.1f}s")

    pprint_routine(result)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[runner] wrote {args.out}")

    if result.get("error"):
        print("[runner] partial run / error surfaced", file=sys.stderr)
        return 2
    if not result.get("routine"):
        print("[runner] EMPTY routine", file=sys.stderr)
        return 2
    if len(result["routine"]) < args.expect:
        print(
            f"[runner] expected >= {args.expect} products, got {len(result['routine'])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())