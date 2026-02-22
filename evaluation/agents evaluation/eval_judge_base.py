"""
eval_judge_base.py — Shared utilities for all LLM-as-a-Judge evaluators.

Provides:
  - get_judge_llm()       : Returns a deterministic ChatOpenAI instance for judging
  - call_judge()          : Invokes the judge LLM and parses its JSON response
  - extract_judge_trace() : Pulls the [Judge] trace from the messages list
  - extract_all_traces()  : Returns the full formatted trace string
  - make_na_result()      : Standard EvaluationResult for N/A cases

All evaluators import from here so the judge model and endpoint are
configured in exactly one place.

Environment variables (from your .env):
  OPENAI_API_KEY       — required, your llmod.ai / OpenAI key
  EVAL_JUDGE_MODEL     — optional, defaults to "gpt-4o"
  EVAL_JUDGE_BASE_URL  — REQUIRED if using a custom endpoint (e.g. llmod.ai).
                         If omitted, requests go to api.openai.com and will
                         fail with 401 when using a non-OpenAI key.
                         Set to: https://api.llmod.ai/v1
"""

import json
import logging
import os
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Judge LLM
# ─────────────────────────────────────────────────────────────────────────────

def get_judge_llm() -> ChatOpenAI:
    """
    Return a ChatOpenAI-compatible instance for evaluation.

    Uses the SAME model, base_url, and api_key as your clinical system so
    the judge authenticates correctly against llmod.ai.

    Required .env variables:
      OPENAI_API_KEY          — your llmod.ai key
      EVAL_JUDGE_MODEL        — e.g. RPRTHPB-gpt-5-mini
      EVAL_JUDGE_BASE_URL     — https://api.llmod.ai/v1
      EVAL_JUDGE_TEMPERATURE  — optional, defaults to 1 (match your system)
    """
    model    = os.getenv("EVAL_JUDGE_MODEL")
    base_url = os.getenv("EVAL_JUDGE_BASE_URL")
    temp     = float(os.getenv("EVAL_JUDGE_TEMPERATURE", "1"))

    if not model:
        raise EnvironmentError(
            "EVAL_JUDGE_MODEL is not set. "
            "Add it to your .env: EVAL_JUDGE_MODEL=RPRTHPB-gpt-5-mini"
        )
    if not base_url:
        raise EnvironmentError(
            "EVAL_JUDGE_BASE_URL is not set. "
            "Add it to your .env: EVAL_JUDGE_BASE_URL=https://api.llmod.ai/v1"
        )

    log.info("Judge LLM: model=%s  base_url=%s  temperature=%s", model, base_url, temp)

    return ChatOpenAI(
        model=model,
        temperature=temp,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON-safe judge invocation
# ─────────────────────────────────────────────────────────────────────────────

def call_judge(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> dict:
    """
    Call the judge LLM and parse its JSON response safely.

    The judge is instructed (in each evaluator's system prompt) to return
    ONLY a JSON object. This helper strips accidental markdown fences and
    returns an empty dict on any parse failure so the pipeline never crashes.
    """
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()
        # Strip accidental ```json ... ``` fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Judge JSON parse error: %s | raw=%s", e, raw[:200] if "raw" in dir() else "")
        return {}
    except Exception as e:
        log.error("Judge LLM call failed: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Trace extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_judge_trace(messages: list) -> str:
    """
    Find and return the [Judge] system trace message from the messages list.
    Returns a fallback string if not found.
    """
    for m in messages:
        if m.get("role") == "system" and "[Judge]" in m.get("content", ""):
            return m["content"]
    return "No [Judge] trace found in messages."


def extract_all_traces(messages: list, max_chars: int = 1500) -> str:
    """
    Format ALL messages in the state as a readable trace string for the judge.
    Truncated to max_chars to stay within token budgets.
    """
    lines = []
    for m in messages:
        role    = m.get("role", "unknown").upper()
        content = m.get("content", "")
        lines.append(f"[{role}] {content}")
    full = "\n".join(lines)
    return full[:max_chars] if len(full) > max_chars else full


# ─────────────────────────────────────────────────────────────────────────────
# Standard N/A result
# ─────────────────────────────────────────────────────────────────────────────

def make_na_result(key: str, reason: str) -> EvaluationResult:
    """Return a standard N/A EvaluationResult for inapplicable evaluators."""
    return EvaluationResult(
        key=key,
        score=None,
        comment=f"N/A — {reason}",
    )