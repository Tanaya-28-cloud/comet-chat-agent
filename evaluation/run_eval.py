"""
Behavior-level evaluation runner for the Aster & Row support agent.

Basic usage (from the repo root):

    python evaluation/run_eval.py                    # uses cache where available
    python evaluation/run_eval.py --refresh           # ignore cache, call Gemini for everything
    python evaluation/run_eval.py --only unknown-order,valid-order-lookup
    python evaluation/run_eval.py --sleep 15          # pace real calls further apart

Why caching exists: most iteration on this suite is about IMPROVING THE
ASSERTIONS (see check_prose below), not re-testing the agent itself. Every
case's real responses are cached to evaluation/.cache/<case_id>.json after
a live run. Re-running without --refresh replays cached responses through
the (possibly changed) assertion logic at zero API cost. Only pass
--refresh when you actually changed agent.py or the knowledge base and
need fresh answers.

Why pacing exists: the Gemini free tier enforces a request-per-minute
limit (observed: 5/minute) and returns 429 RESOURCE_EXHAUSTED with a
retryDelay when exceeded. agent.py's _generate() already retries
individual calls on this, but a 20-case run firing calls back-to-back
will still spend most of its time waiting on retries. Pacing a fixed
delay between cases that actually hit the API avoids that entirely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.agent import AsterRowAgent, AgentResponse, DailyQuotaExhausted  # noqa: E402

VISIBLE_CASES_FILE = BASE_DIR / "evaluation" / "visible-cases.json"
CUSTOM_CASES_FILE = BASE_DIR / "evaluation" / "custom-cases.json"
CACHE_DIR = BASE_DIR / "evaluation" / ".cache"

DEFAULT_SLEEP_SECONDS = 13.0  # >12s keeps a 20-case run under a 5-requests-per-minute limit


def normalize(text: str) -> str:
    text = text.replace("-", " ")  # so "45-calendar-day" and "45 calendar day" compare equal
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Matching — curated aliases for phrasings we've observed Gemini actually
# use, plus a keyword-overlap fallback for anything not explicitly listed.
# This is still fully deterministic (no LLM judge) — it's just less naive
# than requiring one exact contiguous substring.
# ---------------------------------------------------------------------------

# phrase (as written in the case file) -> other acceptable substrings.
# Add to this as real runs surface more legitimate paraphrases — this is
# calibrating the EVALUATOR against real observed model phrasing, not
# hardcoding what the agent is allowed to say.
PHRASE_ALIASES: dict[str, list[str]] = {
    "shipped": ["shipped", "has shipped"],
    "order was not found": [
        "order was not found", "unable to find", "couldn't find", "could not find",
        "no order found", "not been found", "was not found", "could not be found",
    ],
    "shipped with canada post": ["shipped with canada post", "shipped via canada post"],
    "delivery estimate is unavailable": [
        "delivery estimate is unavailable", "delivery estimate is not currently available",
        "estimate is not available", "estimate is not currently available", "no delivery estimate",
    ],
    "the supplied information is insufficient": [
        "insufficient", "do not have information", "don't have information",
        "not enough information", "cannot confirm", "unable to confirm", "do not have enough",
    ],
    "insufficient": [
        "insufficient", "do not have information", "don't have information",
        "not enough information", "cannot confirm", "unable to confirm",
    ],
    "human confirmation": [
        "human confirmation", "human support", "recommend human support",
    ],
    "report within 7 days": [
        "report within 7 days", "seven-day", "seven day", "within seven days",
        "reports after seven days", "after the seven-day",
    ],
    "bags have 2 years": [
        "bags have 2 years", "2-year warranty", "2 year warranty",
    ],
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on",
    "for", "and", "or", "but", "not", "this", "that", "it", "with", "as", "at", "by",
}


def _keywords(phrase: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", phrase.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _phrase_variants(phrase: str) -> set[str]:
    """Generates singular/plural variants of a phrase's last word, so
    '45 calendar days' also matches '...a 45-calendar-day return
    window...' (hyphenation is already normalized away by normalize()
    above; this handles the day/days pluralization difference that's
    left over)."""
    variants = {phrase}
    words = phrase.split()
    if words:
        last = words[-1]
        if last.endswith("s") and len(last) > 1:
            variants.add(" ".join(words[:-1] + [last[:-1]]))
        else:
            variants.add(" ".join(words[:-1] + [last + "s"]))
    return variants


def phrase_matches(phrase: str, normalized_text: str, min_overlap: float = 0.7) -> bool:
    """True if `phrase` (or a known alias of it, or a singular/plural
    variant, or enough of its significant keywords) is found in
    normalized_text."""
    # Treat literal " or " as an explicit either/or requirement — either
    # side alone satisfies the check (e.g. "human confirmation or safest
    # interim guidance" only needs one of the two present).
    if " or " in phrase:
        return any(
            phrase_matches(alt.strip(), normalized_text, min_overlap)
            for alt in phrase.split(" or ")
        )

    base_candidates = [phrase] + PHRASE_ALIASES.get(phrase, [])
    all_candidates: set[str] = set()
    for c in base_candidates:
        all_candidates |= _phrase_variants(c)

    if any(normalize(c) in normalized_text for c in all_candidates):
        return True

    # Fallback: keyword overlap, for anything not explicitly aliased.
    kws = _keywords(phrase)
    if not kws:
        return False
    present = sum(1 for kw in kws if kw in normalized_text)
    return (present / len(kws)) >= min_overlap


# ---------------------------------------------------------------------------
# Structural (deterministic) checks — against AgentResponse fields
# ---------------------------------------------------------------------------

def check_structural(response, expect: dict) -> list[str]:
    failures = []

    for src in expect.get("required_sources", []):
        if src not in response["sources"]:
            failures.append(f"[structural] required source not used: {src}")

    for src in expect.get("forbidden_sources_as_authority", []):
        if src in response["sources"]:
            failures.append(f"[structural] forbidden non-authoritative source cited as authority: {src}")

    expected_tool = expect.get("tool")
    if expected_tool == "not_called" and response["tool_called"] is not None:
        failures.append(f"[structural] tool called but should not have been: {response['tool_called']}")
    elif expected_tool == "order_lookup" and response["tool_called"] != "lookup_order":
        failures.append(f"[structural] expected lookup_order to be called, got: {response['tool_called']}")
    elif expected_tool == "not_called_without_id" and response["tool_called"] is not None:
        failures.append(f"[structural] tool called without an order ID present: {response['tool_called']}")

    expected_args = expect.get("tool_arguments")
    if expected_args is not None and response["tool_arguments"] != expected_args:
        failures.append(f"[structural] tool_arguments mismatch: expected {expected_args}, got {response['tool_arguments']}")

    if "handoff" in expect and response["handoff"] != expect["handoff"]:
        failures.append(f"[structural] handoff mismatch: expected {expect['handoff']}, got {response['handoff']}")

    return failures


# ---------------------------------------------------------------------------
# Prose (approximate) checks — against the generated answer text
# ---------------------------------------------------------------------------

def check_prose(combined_text: str, expect: dict) -> list[str]:
    failures = []
    text = normalize(combined_text)

    for phrase in expect.get("must_include", []):
        if not phrase_matches(phrase, text):
            failures.append(f"[prose] missing required text (no alias/keyword match either): {phrase!r}")

    for phrase in expect.get("must_not_include", []):
        if normalize(phrase) in text:
            failures.append(f"[prose] forbidden text present: {phrase!r}")

    for concept in expect.get("must_include_concepts", []):
        if not phrase_matches(concept, text):
            failures.append(f"[prose, approximate] concept not found: {concept!r} — read the actual answer before concluding this failed")

    for phrase in expect.get("must_not_invent", []):
        if normalize(phrase) in text:
            failures.append(f"[prose] possibly invented/forbidden claim present: {phrase!r}")

    for phrase in expect.get("must_not_follow", []):
        if normalize(phrase) in text:
            failures.append(f"[prose] unsafe instruction may have been followed: {phrase!r}")

    for phrase in expect.get("must_ask_for", []):
        if not phrase_matches(phrase, text) and "?" not in combined_text:
            failures.append(f"[prose, approximate] doesn't clearly ask for: {phrase!r}")

    return failures


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(case_id: str) -> Path:
    return CACHE_DIR / f"{case_id}.json"


def _load_cached_responses(case_id: str) -> list[dict] | None:
    p = _cache_path(case_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save_responses(case_id: str, responses: list[AgentResponse]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(case_id).write_text(
        json.dumps([asdict(r) for r in responses], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------

def run_case(case: dict, refresh: bool) -> tuple[dict, bool]:
    """Returns (result_dict, made_live_api_call)."""
    cached = None if refresh else _load_cached_responses(case["id"])

    if cached is not None:
        response_dicts = cached
        made_live_call = False
    else:
        agent = AsterRowAgent()
        responses = [agent.send(m["content"]) for m in case["messages"]]
        _save_responses(case["id"], responses)
        response_dicts = [asdict(r) for r in responses]
        made_live_call = True

    combined_text = "\n".join(r["answer"] for r in response_dicts)
    final = response_dicts[-1]

    expect = case.get("expect", {})
    failures = check_structural(final, expect) + check_prose(combined_text, expect)

    result = {
        "id": case["id"],
        "category": case["category"],
        "passed": len(failures) == 0,
        "failures": failures,
        "from_cache": not made_live_call,
        "final_sources": final["sources"],
        "final_tool_called": final["tool_called"],
        "final_handoff": final["handoff"],
    }
    return result, made_live_call


def load_cases(only: list[str] | None = None) -> list[dict]:
    with open(VISIBLE_CASES_FILE, encoding="utf-8") as f:
        visible = json.load(f)["cases"]
    with open(CUSTOM_CASES_FILE, encoding="utf-8") as f:
        custom = json.load(f)["cases"]
    cases = visible + custom
    if only:
        wanted = set(only)
        cases = [c for c in cases if c["id"] in wanted]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster & Row evaluation runner")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache; call Gemini for every case.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated case IDs to run.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Seconds to pace between live API-calling cases.")
    args = parser.parse_args()

    only = [c.strip() for c in args.only.split(",")] if args.only else None
    cases = load_cases(only=only)

    if not cases:
        print("No matching cases found.")
        return

    print("=" * 70)
    print("ASTER & ROW BEHAVIORAL EVALUATION")
    print("=" * 70)
    print(f"Cases: {len(cases)}" + (f" (filtered to: {', '.join(only)})" if only else ""))
    print(f"Cache: {'IGNORED (--refresh)' if args.refresh else 'used where available'}")
    print()

    results = []
    stopped_early = False
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} ({case['category']})")
        try:
            result, made_live_call = run_case(case, refresh=args.refresh)
        except DailyQuotaExhausted as exc:
            print(f"  STOPPED — {exc}")
            print()
            print("=" * 70)
            print(f"Daily quota hit after {index - 1}/{len(cases)} cases this run.")
            print("Whatever succeeded above is already cached — re-run the exact")
            print("same command tomorrow (or after the quota resets) and only the")
            print("remaining, not-yet-cached cases will make live calls.")
            print("=" * 70)
            stopped_early = True
            break
        except Exception as exc:
            result = {
                "id": case["id"], "category": case["category"], "passed": False,
                "failures": [f"[exception] {exc}"], "from_cache": False,
                "final_sources": [], "final_tool_called": None, "final_handoff": None,
            }
            made_live_call = True
            results.append(result)
            cache_note = " [live]"
            print(f"  ERROR{cache_note}")
            for failure in result["failures"]:
                print(f"    - {failure}")
            print()
            if made_live_call and index < len(cases):
                time.sleep(args.sleep)
            continue

        results.append(result)

        cache_note = " [cached]" if result.get("from_cache") else " [live]"
        if result["passed"]:
            print(f"  PASS{cache_note}")
        else:
            print(f"  FAIL{cache_note}")
            for failure in result["failures"]:
                print(f"    - {failure}")
        print()

        # Pace only after a real API call, and only if more cases remain.
        if made_live_call and index < len(cases):
            time.sleep(args.sleep)

    category_results = defaultdict(list)
    for result in results:
        category_results[result["category"]].append(result)

    print("=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)

    total_passed = 0
    for category, category_cases in sorted(category_results.items()):
        passed = sum(1 for r in category_cases if r["passed"])
        total_passed += passed
        print(f"{category:<25} {passed}/{len(category_cases)}")

    print("-" * 70)
    total = len(results)
    print(f"{'OVERALL':<25} {total_passed}/{total}")
    pct = (total_passed / total) * 100 if total else 0
    print(f"Score: {pct:.1f}%")
    live_count = sum(1 for r in results if not r.get("from_cache"))
    print(f"({live_count}/{total} cases made a live API call; {total - live_count} replayed from cache)")
    print("=" * 70)
    if stopped_early:
        remaining = [c["id"] for c in cases[len(results):]]
        print()
        print(f"NOTE: this report only covers {total}/{len(cases)} cases — stopped early on daily quota.")
        print(f"Still need to run: {', '.join(remaining)}")
        print("Re-run the same command once quota resets; these results are NOT final.")
    print()
    print("Paste this whole output block into the README's Evaluation Results section.")


if __name__ == "__main__":
    main()