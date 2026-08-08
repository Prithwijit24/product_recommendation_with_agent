"""Research sub-agents: skin-concern inference and ingredient shortlisting.

Each is a small worker-role LLM loop over aistack search/crawl evidence.
"""

from __future__ import annotations

import json

from . import aistack, providers

SKIN_CONCERNS_SYSTEM = """You are a dermatology research sub-agent.
Profile: {profile}

Find the 2-4 most likely skin-science considerations implied by this profile using
credible dermatology/health sources only. Ground your reasoning in skin biology
(melanin density, sun-sensitivity, photoaging, hormonal changes) — never race labels.
Output ONLY a JSON array (no prose):
[{{"concern": "...", "evidence": "one-line", "source": "url"}}]
"""

INGREDIENTS_SYSTEM = """You are a cosmetic science research sub-agent.
Concerns to address: {concerns}

Research cosmetic ingredients that may help with each concern. Do not include
medical treatments or prescription claims. Output ONLY a JSON array (no prose):
[{{"ingredient": "...", "addresses": "concern", "why": "one-line mechanism"}}]
Max 5 ingredients.
"""


def _gather_evidence(client, query: str, rounds: int = 2) -> str:
    chunks = []
    seen: set[str] = set()
    try:
        for r in range(rounds):
            results = client.search(query, max_results=3)
            for res in results:
                url = res.get("url", "")
                if url in seen or len(chunks) >= 6:
                    continue
                seen.add(url)
                chunks.append(f"- {res.get('title','')}: {res.get('content','')[:300]}")
    except Exception:
        return "- no evidence gathered"
    return "\n".join(chunks) or "- no evidence gathered"


def _clean_and_parse(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text.replace("```", "")
        text = text.replace("```", "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _run_loop(system: str, user: str, client: aistack.AistackClient | None, worker=providers.call_llm) -> list[dict]:
    client = client or aistack.AistackClient()
    evidence = _gather_evidence(client, user, rounds=2)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "user", "content": f"Evidence from search:\n{evidence}\nNow respond strictly in JSON."},
    ]
    try:
        resp = worker(messages, tools=None, role="worker")
    except Exception:
        return []
    return _clean_and_parse(resp.get("content") or "")


def research_skin_concerns(
    profile_text: str, worker=providers.call_llm, client: aistack.AistackClient | None = None
) -> list[dict]:
    system = SKIN_CONCERNS_SYSTEM.format(profile=profile_text)
    return _run_loop(system, f"Profile: {profile_text}. Research and output the JSON now.", client, worker)


def research_ingredients(
    concerns: list[str],
    worker=providers.call_llm,
    client: aistack.AistackClient | None = None,
) -> list[dict]:
    joined = ", ".join(concerns)
    system = INGREDIENTS_SYSTEM.format(concerns=joined)
    return _run_loop(system, f"Concerns: {joined}. Output the JSON now.", client, worker)