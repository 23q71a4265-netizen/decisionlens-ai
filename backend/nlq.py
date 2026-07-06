"""
DecisionLens AI - Natural Language Query Layer
------------------------------------------------
Two tiers, so the product works out of the box AND gets smarter if you
add an API key:

  1. RULE-BASED (always on, zero cost, zero external calls): parses the
     question for a target column + intent ("why did X drop", "forecast X",
     "what's driving X") and answers using the real analytics.py functions.

  2. LLM-POWERED (optional): if ANTHROPIC_API_KEY is set, the computed
     stats from tier 1 are handed to Claude to turn into a fluent,
     decision-ready recommendation instead of a templated sentence.
     This is a "hand real numbers to the model, don't let it guess"
     pattern - Claude never sees raw uploaded data, only the aggregated
     stats we already computed, which keeps this cheap and accurate.
"""

from __future__ import annotations

import os
import re
import json
from typing import Optional

import pandas as pd

from analytics import correlation_drivers, dataset_summary, linear_forecast, threshold_alerts


def _closest_column(question: str, columns: list[str]) -> Optional[str]:
    q = question.lower()
    # exact / substring match first
    for col in columns:
        if col.lower() in q:
            return col
    # loose match on tokens (e.g. "sale" -> "sales")
    tokens = re.findall(r"[a-zA-Z]+", q)
    for col in columns:
        col_l = col.lower()
        for t in tokens:
            if len(t) > 3 and (t in col_l or col_l in t):
                return col
    return None


def answer_question(df: pd.DataFrame, question: str, group_col: Optional[str] = None) -> dict:
    from analytics import numeric_columns
    cols = numeric_columns(df)
    target = _closest_column(question, cols) or (cols[0] if cols else None)

    if target is None:
        return {"answer": "No numeric columns found in this dataset to analyze.", "facts": {}}

    q = question.lower()
    facts: dict = {"target": target}

    if any(w in q for w in ["why", "drop", "fell", "decline", "drove", "driving", "cause"]):
        drivers = correlation_drivers(df, target)
        alerts = threshold_alerts(df, group_col=group_col)
        relevant_alerts = [a for a in alerts if a["column"] == target or a["column"] in [d["column"] for d in drivers]]
        facts["drivers"] = drivers
        facts["alerts"] = relevant_alerts
        answer = _rule_based_why(target, drivers, relevant_alerts)

    elif any(w in q for w in ["forecast", "predict", "next", "will", "expect"]):
        periods_match = re.search(r"(\d+)\s*(day|week|period)", q)
        periods = int(periods_match.group(1)) if periods_match else 7
        fc = linear_forecast(df[target], periods=periods)
        facts["forecast"] = fc
        answer = _rule_based_forecast(target, fc, periods)

    else:
        summ = dataset_summary(df)["metrics"].get(target, {})
        drivers = correlation_drivers(df, target, top_n=3)
        facts["summary"] = summ
        facts["drivers"] = drivers
        answer = _rule_based_summary(target, summ, drivers)

    llm_answer = _try_llm_explain(question, facts)
    return {
        "answer": llm_answer or answer,
        "facts": facts,
        "powered_by": "claude" if llm_answer else "rule-engine",
    }


def _rule_based_why(target: str, drivers: list[dict], alerts: list[dict]) -> str:
    if not drivers:
        return f"'{target}' has no strongly correlated columns in this dataset yet — add more signals (inventory, pricing, weather, traffic) to get a causal read."
    lead = drivers[0]
    parts = [
        f"'{target}' is most strongly correlated with '{lead['column']}' "
        f"({lead['relationship']}, {lead['strength']} correlation of {lead['correlation']})."
    ]
    if len(drivers) > 1:
        others = ", ".join(f"{d['column']} ({d['correlation']})" for d in drivers[1:3])
        parts.append(f"Other contributing factors: {others}.")
    if alerts:
        sev = alerts[0]
        parts.append(
            f"Live alert: {sev['column']} is showing a {sev['direction']} "
            f"(z-score {sev['z_score']}) in group '{sev['group']}' — this is likely the active driver right now."
        )
    return " ".join(parts)


def _rule_based_forecast(target: str, fc: dict, periods: int) -> str:
    if "error" in fc:
        return fc["error"]
    direction = fc["trend_direction"]
    start, end = fc["forecast"][0], fc["forecast"][-1]
    return (
        f"'{target}' is trending {direction} (slope {fc['trend_slope']}/period). "
        f"Over the next {periods} periods, expect values moving from ~{start} to ~{end}, "
        f"with a 95% range of {fc['lower_bound'][-1]} to {fc['upper_bound'][-1]} by the end of the window."
    )


def _rule_based_summary(target: str, summ: dict, drivers: list[dict]) -> str:
    if not summ:
        return f"No data available yet for '{target}'."
    change = summ.get("pct_change_latest", 0)
    direction = "up" if change > 0 else "down" if change < 0 else "flat"
    driver_txt = ""
    if drivers:
        driver_txt = f" Most correlated with '{drivers[0]['column']}' ({drivers[0]['correlation']})."
    return (
        f"'{target}' latest value is {summ['latest']} ({direction} {abs(change)}% vs prior point). "
        f"Range so far: {summ['min']}–{summ['max']}, average {summ['mean']}.{driver_txt}"
    )


def _try_llm_explain(question: str, facts: dict) -> Optional[str]:
    """Optional Claude-powered explanation. Silently falls back if no key set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # pip install anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are DecisionLens AI, a data intelligence copilot. A user asked:\n"
            f"\"{question}\"\n\n"
            "Here are the pre-computed statistics (already correct — do not invent new numbers, "
            "only interpret these):\n"
            f"{json.dumps(facts, default=str, indent=2)}\n\n"
            "In 2-4 sentences, give a decision-ready answer: state what happened, the most likely "
            "cause from the data, and one concrete recommended action. No preamble."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if hasattr(block, "text")).strip()
    except Exception:
        # Any failure (bad key, network, rate limit) -> silently use rule-based answer
        return None
