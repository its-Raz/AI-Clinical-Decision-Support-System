"""
backend/agents/blood_test_analyst/run_batch.py

Batch adapter for the Blood Test Analyst ReAct agent.

ROOT CAUSE NOTE — "state module" bug:
  The react_agent was built to run as a standalone script, so it uses
  bare imports: `from state import ...`, `from nodes import ...` etc.
  When called as part of a package these bare names are not on sys.path.
  Fix: we temporarily inject the react_agent directory into sys.path
  before importing, then remove it to keep the environment clean.
"""

import sys
import os
import logging

log = logging.getLogger(__name__)

# ── sys.path fix — must happen BEFORE importing run_react_agent ───────────
_REACT_AGENT_DIR = os.path.join(os.path.dirname(__file__), "react_agent")

def _ensure_react_agent_on_path():
    """Add react_agent/ to sys.path so its bare imports resolve correctly."""
    if _REACT_AGENT_DIR not in sys.path:
        sys.path.insert(0, _REACT_AGENT_DIR)
        log.debug("sys.path: injected %s", _REACT_AGENT_DIR)
        print(f"   🔧 [run_batch] sys.path fix: injected react_agent dir")
    else:
        log.debug("sys.path: react_agent dir already present")

_ensure_react_agent_on_path()

# DO NOT import run_react_agent at module level — it loads the RAG system
# which initializes heavy ML models (embeddings, TensorFlow).
# Import lazily inside the function instead.

# Flags that warrant deep analysis
_ABNORMAL_FLAGS = {"low", "high", "critical_low", "critical_high"}


def run_batch_analyst(state: dict) -> dict:
    """
    LangGraph node — wraps run_react_agent for a batch of lab results.

    Reads:   state["lab_result"]  (List[dict])
    Writes:  state["lab_insights"] (aggregated summaries, str)
             state["messages"]    (trace entries)
    """
    log.debug("run_batch_analyst() called")
    print("\n" + "─" * 50)
    print("🔬 [run_batch_analyst] ENTER")

    # ── Lazy import — only load RAG system when actually needed ───────
    from backend.agents.blood_test_analyst.react_agent.run import run_react_agent

    patient_id  = state.get("patient_id", "unknown")
    lab_results = state.get("lab_result") or []

    print(f"   patient_id  : {patient_id}")
    print(f"   total batch : {len(lab_results)} metrics")
    log.info("run_batch_analyst: patient=%s, batch_size=%d", patient_id, len(lab_results))

    # ── Debug: print every metric in the batch ────────────────────────
    for i, r in enumerate(lab_results):
        flag = r.get("flag", "normal")
        print(f"   [{i}] {r.get('test_name','?')} = {r.get('value','?')} "
              f"{r.get('unit','')} | flag={flag}")

    trace_msgs = []

    # ── Filter to abnormal results only ──────────────────────────────
    to_analyse = [
        r for r in lab_results
        if r.get("flag", "normal") in _ABNORMAL_FLAGS
    ]
    normal_count   = len(lab_results) - len(to_analyse)
    abnormal_count = len(to_analyse)

    print(f"   abnormal    : {abnormal_count}  |  normal (skipped): {normal_count}")
    log.info("run_batch_analyst: %d abnormal, %d normal", abnormal_count, normal_count)

    trace_msgs.append({
        "role":    "system",
        "content": (
            f"[Blood Test Analyst] Received batch for {patient_id}. "
            f"{abnormal_count} abnormal / {normal_count} normal metrics."
        ),
    })

    if not to_analyse:
        msg = (
            "All laboratory values for this panel are within normal reference "
            "ranges. No clinical intervention is indicated at this time."
        )
        print("   ✅ All values normal — skipping RAG analysis")
        trace_msgs.append({
            "role":    "system",
            "content": "[Blood Test Analyst] All values normal — no RAG required.",
        })
        print("─" * 50)
        return {"lab_insights": msg, "messages": trace_msgs}

    # ── Run ReAct agent for each abnormal result ──────────────────────
    summaries = []

    for idx, result in enumerate(to_analyse):
        test_name = result["test_name"]
        print(f"\n   🧪 [{idx+1}/{abnormal_count}] Analysing: {test_name} "
              f"= {result.get('value')} {result.get('unit')} (flag={result.get('flag')})")
        log.info("run_batch_analyst: analysing %s", test_name)

        trace_msgs.append({
            "role":    "system",
            "content": (
                f"[Blood Test Analyst → ReAct] Analysing {test_name} "
                f"({result['value']} {result['unit']}, flag={result['flag']}) …"
            ),
        })

        mini_state = {
            "patient_id": patient_id,
            "lab_result": result,
        }

        try:
            result_state = run_react_agent(mini_state)
            summary      = result_state.get("react_summary", "")

            if summary:
                print(f"   ✅ Summary ready: {len(summary)} chars")
                log.info("run_batch_analyst: %s summary=%d chars", test_name, len(summary))
                summaries.append(f"**{test_name}**\n{summary}")
                trace_msgs.append({
                    "role":    "system",
                    "content": f"[Blood Test Analyst → ReAct] {test_name} summary ready ({len(summary)} chars).",
                })
            else:
                print(f"   ⚠️  Empty summary returned for {test_name}")
                log.warning("run_batch_analyst: empty summary for %s", test_name)
                trace_msgs.append({
                    "role":    "system",
                    "content": f"[Blood Test Analyst] ⚠️ Empty summary for {test_name}.",
                })

        except Exception as e:
            log.error("run_batch_analyst: error analysing %s: %s", test_name, e, exc_info=True)
            print(f"   ❌ ERROR analysing {test_name}: {e}")
            trace_msgs.append({
                "role":    "system",
                "content": f"[Blood Test Analyst] ❌ Error analysing {test_name}: {e}",
            })

    # ── Aggregate ─────────────────────────────────────────────────────
    aggregated = "\n\n---\n\n".join(summaries) if summaries else "No significant findings."

    print(f"\n   📋 Aggregated insights: {len(aggregated)} chars from {len(summaries)} summaries")
    log.info("run_batch_analyst: done. %d summaries aggregated", len(summaries))

    trace_msgs.append({
        "role":    "system",
        "content": (
            f"[Blood Test Analyst] Batch complete. "
            f"{len(summaries)}/{abnormal_count} results analysed successfully."
        ),
    })

    print("─" * 50)
    return {
        "lab_insights": aggregated,
        "messages":     trace_msgs,
    }