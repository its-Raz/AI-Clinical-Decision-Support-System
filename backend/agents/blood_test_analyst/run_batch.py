"""
backend/agents/blood_test_analyst/run_batch.py

Batch adapter for the Blood Test Analyst ReAct agent.

REFACTOR NOTE — single ReAct run:
  Previously this looped over each abnormal metric and called run_react_agent
  separately, causing get_patient_history to be called N times for the same
  patient and N separate summary LLM calls.

  Now a single run_react_agent call receives ALL abnormal metrics at once.
  The ReAct agent calls get_patient_history once, check_reference_range once
  per metric (unavoidable — values differ), and produces one consolidated
  summary — significantly more efficient.

ROOT CAUSE NOTE — "state module" bug:
  The react_agent was built to run as a standalone script, so it uses
  bare imports: `from state import ...`, `from nodes import ...` etc.
  When called as part of a package these bare names are not on sys.path.
  Fix: inject the react_agent directory into sys.path before importing.
"""

import sys
import os
import logging

log = logging.getLogger(__name__)

_REACT_AGENT_DIR = os.path.join(os.path.dirname(__file__), "react_agent")


def _ensure_react_agent_on_path():
    if _REACT_AGENT_DIR not in sys.path:
        sys.path.insert(0, _REACT_AGENT_DIR)
        log.debug("sys.path: injected %s", _REACT_AGENT_DIR)
        print(f"   🔧 [run_batch] sys.path fix: injected react_agent dir")


_ensure_react_agent_on_path()

_ABNORMAL_FLAGS = {"low", "high", "critical_low", "critical_high"}


def run_batch_analyst(state: dict) -> dict:
    """
    LangGraph node — runs a SINGLE ReAct agent pass for all abnormal metrics.

    Reads:   state["lab_result"]   (List[dict])
    Writes:  state["lab_insights"] (consolidated summary, str)
             state["messages"]     (trace entries)
             state["steps"]        (structured step objects for API trace)
    """
    log.debug("run_batch_analyst() called")
    print("\n" + "─" * 50)
    print("🔬 [run_batch_analyst] ENTER")

    from backend.agents.blood_test_analyst.react_agent.run import run_react_agent

    patient_id  = state.get("patient_id", "unknown")
    lab_results = state.get("lab_result") or []

    print(f"   patient_id  : {patient_id}")
    print(f"   total batch : {len(lab_results)} metrics")

    print(f"\n   📊 FULL BATCH BREAKDOWN:")
    for i, r in enumerate(lab_results):
        flag        = r.get("flag", "normal")
        is_abnormal = flag in _ABNORMAL_FLAGS
        marker      = "🔴" if is_abnormal else "✅"
        print(f"   [{i}] {marker} {r.get('test_name', '?'):20} = {r.get('value', '?'):>6} "
              f"{r.get('unit', ''):6} | flag={flag:15} | analyze={is_abnormal}")

    trace_msgs = []

    to_analyse     = [r for r in lab_results if r.get("flag", "normal") in _ABNORMAL_FLAGS]
    normal_count   = len(lab_results) - len(to_analyse)
    abnormal_count = len(to_analyse)

    print(f"   abnormal    : {abnormal_count}  |  normal (skipped): {normal_count}")

    trace_msgs.append({
        "role":    "system",
        "content": (
            f"[Blood Test Analyst] Received batch for {patient_id}. "
            f"{abnormal_count} abnormal / {normal_count} normal metrics."
        ),
    })

    # ── Short-circuit: all values normal ─────────────────────────────────
    if not to_analyse:
        msg = (
            "All laboratory values for this panel are within normal reference "
            "ranges. No clinical intervention is indicated at this time."
        )
        print("   ✅ All values normal — skipping ReAct analysis")
        all_steps = [{
            "module":   "BloodTestAnalyst",
            "prompt":   f"Patient: {patient_id}\nLab results: {lab_results}",
            "response": msg,
        }]
        print("─" * 50)
        return {"lab_insights": msg, "messages": trace_msgs, "steps": all_steps}

    # ── Single ReAct run for ALL abnormal metrics ─────────────────────────
    metric_names = [r["test_name"] for r in to_analyse]
    print(f"\n   🧪 Single ReAct pass for {abnormal_count} metric(s): {metric_names}")

    trace_msgs.append({
        "role":    "system",
        "content": (
            f"[Blood Test Analyst → ReAct] Single-pass analysis for: "
            f"{', '.join(metric_names)} …"
        ),
    })

    # lab_result is now a LIST — run.py detects this and formats accordingly
    mini_state = {
        "patient_id": patient_id,
        "lab_result": to_analyse,
    }

    try:
        result_state = run_react_agent(mini_state)
        summary      = result_state.get("lab_insights", "")

        if not summary:
            summary = "No significant findings."
            log.warning("run_batch_analyst: empty summary returned")

        all_steps = result_state.get("steps", [])
        print(f"   ✅ Consolidated summary: {len(summary)} chars | steps: {len(all_steps)}")

        trace_msgs.append({
            "role":    "system",
            "content": f"[Blood Test Analyst] Single-pass complete ({len(summary)} chars).",
        })

    except Exception as e:
        log.error("run_batch_analyst: error — %s", e, exc_info=True)
        print(f"   ❌ ERROR: {e}")
        summary   = f"Analysis failed: {e}"
        all_steps = [{
            "module":   "BloodTestAnalyst/ReAct",
            "prompt":   f"Analyze {metric_names}",
            "response": f"ERROR: {e}",
        }]
        trace_msgs.append({
            "role":    "system",
            "content": f"[Blood Test Analyst] ❌ Error: {e}",
        })

    print(f"\n   📄 PREVIEW: {summary[:300]}...")
    print("─" * 50)

    return {
        "lab_insights": summary,
        "messages":     trace_msgs,
        "steps":        all_steps,
    }