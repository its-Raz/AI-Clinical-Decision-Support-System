"""
backend/agents/evidence_analyst/run.py

Run function for the Evidence Analyst ReAct agent.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from backend.agents.evidence_analyst.__init__ import ReActAgent
from backend.agents.evidence_analyst.state import ReActInternalState

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.agents.evidence_analyst.prompts import REACT_SYSTEM_PROMPT
from backend.agents.evidence_analyst.utils import extract_latest_user_query


def _get_query(global_state: dict) -> str:
    """
    Resolve the user's query from global_state.

    Priority order:
      1. raw_user_input  — set by app.py / api.py before the graph runs
      2. extract_latest_user_query — fallback for standalone / test invocations
         that build a messages list instead of using AgentState fields

    This dual approach means the evidence analyst works correctly whether it
    is called from the main pipeline (AgentState with raw_user_input) or from
    a standalone test script (dict with a messages list).
    """
    # ── Primary: read the AgentState field directly ────────────────────
    raw = global_state.get("raw_user_input", "").strip()
    if raw:
        return raw

    # ── Fallback: try to extract from messages list ────────────────────
    fallback = extract_latest_user_query(global_state)
    if fallback and "Could not find" not in fallback:
        return fallback

    # ── Last resort: lab_result description (for batch-style calls) ────
    lab = global_state.get("lab_result")
    if lab and isinstance(lab, dict):
        return (
            f"Analyze this lab result for patient {global_state.get('patient_id', '?')}: "
            f"{lab.get('test_name')} = {lab.get('value')} {lab.get('unit')} "
            f"(flag: {lab.get('flag', 'N/A')})"
        )

    return "Please provide a medical question."


def _extract_react_steps(final_state: dict, query: str) -> list[dict]:
    """
    Convert the ReAct agent's internal messages + tool_calls_history
    into structured step objects for the API trace.

    Produces:
      - One "EvidenceAnalyst/ReAct" step per AIMessage that has text content
        (empty AIMessages are pure tool-call triggers — skipped)
      - One "EvidenceAnalyst/Tool:<name>" step per entry in tool_calls_history
    """
    steps = []
    messages           = final_state.get("messages", [])
    tool_calls_history = final_state.get("tool_calls_history", [])

    # ── ReAct LLM steps ───────────────────────────────────────────────
    last_human_content = query

    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type in ("HumanMessage", "ToolMessage"):
            last_human_content = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )

        elif msg_type == "AIMessage":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                steps.append({
                    "module":   "EvidenceAnalyst/ReAct",
                    "prompt":   last_human_content,
                    "response": content,
                })

    # ── Tool call steps ───────────────────────────────────────────────
    for tc in tool_calls_history:
        steps.append({
            "module":   f"EvidenceAnalyst/Tool:{tc.get('tool', 'unknown')}",
            "prompt":   f"tool: {tc.get('tool')}\nargs: {tc.get('args', {})}",
            "response": str(tc.get("result", "")),
        })

    return steps


def run_react_agent(global_state: dict) -> dict:
    """
    Run the Evidence Analyst ReAct agent on a clinical query.

    Reads from global_state:
        raw_user_input       — the user's free-text question (primary source)
        patient_id           — patient identifier

    Writes to global_state:
        evidence_insights    — final answer from the ReAct agent
        tool_calls_history   — all tool calls made during the run
        steps                — structured step objects for the API trace
                               (operator.add will append these to graph-level steps)

    Returns:
        Updated global_state dict
    """
    print("\n" + "=" * 60)
    print("🔬 EVIDENCE ANALYST REACT AGENT: Starting")
    print("=" * 60)

    patient_id = global_state.get("patient_id", "unknown")

    # ── Resolve query — fix for "Could not find a user query" bug ─────
    query = _get_query(global_state)

    print(f"   Patient : {patient_id}")
    print(f"   Query   : {query}")

    # ── Build initial ReAct state ─────────────────────────────────────
    inputs: ReActInternalState = {
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ],
        "iterations":         0,
        "tool_calls_history": [],
    }

    # ── Run the ReAct graph ───────────────────────────────────────────
    agent = ReActAgent()

    print("\n🤖 Running ReAct loop …\n")

    final_state = None
    for state in agent.graph.stream(inputs, stream_mode="values"):
        final_state = state

    # ── Extract results ───────────────────────────────────────────────
    final_answer       = final_state["messages"][-1].content
    tool_calls_history = final_state.get("tool_calls_history", [])

    print(f"\n📊 ReAct complete:")
    print(f"   Tool calls : {len(tool_calls_history)}")
    print(f"   Iterations : {final_state.get('iterations', '?')}")
    print(f"   Answer     : {len(final_answer)} chars")

    # ── Build structured steps for API trace ──────────────────────────
    agent_steps = _extract_react_steps(final_state, query)
    print(f"   Steps captured: {len(agent_steps)}")

    # ── Write back to global_state ────────────────────────────────────
    global_state["evidence_insights"]  = final_answer
    global_state["tool_calls_history"] = tool_calls_history
    global_state["steps"]              = agent_steps

    return global_state