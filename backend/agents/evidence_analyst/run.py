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
    into structured step objects for the API trace in chronological order.
    """
    steps = []
    messages = final_state.get("messages", [])
    tool_calls_history = final_state.get("tool_calls_history", [])

    # 1. יוצרים איטרטור שיאפשר לנו למשוך כלים אחד-אחד לפי הסדר
    history_iter = iter(tool_calls_history)

    from backend.agents.evidence_analyst.prompts import REACT_SYSTEM_PROMPT

    # הפרומפט הראשוני
    current_prompt = f"[SYSTEM]\n{REACT_SYSTEM_PROMPT}\n\n[USER]\n{query}"

    for msg in messages:
        msg_type = type(msg).__name__

        # אם זו תצפית מכלי, אנחנו רק מעדכנים את הפרומפט לאיטרציה הבאה של המודל
        if msg_type == "ToolMessage":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            current_prompt = f"[TOOL OBSERVATION]\n{content}"

        # אם זו הודעת מודל, אנחנו שומרים אותה
        elif msg_type == "AIMessage":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            response_text = content.strip()

            # בודקים אם המודל ביקש להפעיל כלים
            has_tools = hasattr(msg, "tool_calls") and msg.tool_calls

            if has_tools:
                tool_desc = ", ".join([f"{tc['name']}({tc['args']})" for tc in msg.tool_calls])
                response_text += f"\n[Tool Call Triggered: {tool_desc}]"

            # למקרה שהמודל החזיר רק Tool Call בלי טקסט חופשי
            if not response_text and has_tools:
                response_text = "[Tool Call Only]"

            if response_text:
                steps.append({
                    "module": "EvidenceAnalyst/ReAct",
                    "prompt": current_prompt,
                    "response": response_text.strip(),
                })

            # --- התיקון הקריטי ---
            # מיד אחרי שהמודל ביקש כלי, אנחנו שולפים את ביצוע הכלי מההיסטוריה ומכניסים ל-Steps!
            if has_tools:
                for _ in msg.tool_calls:
                    try:
                        tc = next(history_iter)
                        res = tc.get("result", "")

                        # אם ה-RAG החזיר את המילון המורחב עם הפרומפטים
                        if isinstance(res, dict) and "rag_sys_prompt" in res:
                            # שלב א': איך ה-LLM הפנימי של ה-RAG עבד
                            steps.append({
                                "module": "EvidenceAnalyst/RAG_LLM",
                                "prompt": f"[SYSTEM]\n{res['rag_sys_prompt']}\n\n[USER]\n{res['rag_user_prompt']}",
                                "response": res.get("answer", "")
                            })

                            # שלב ב': הכלי עצמו כפי שהוא מוחזר לסוכן
                            steps.append({
                                "module": f"EvidenceAnalyst/Tool:{tc.get('tool', 'unknown')}",
                                "prompt": f"[TOOL ARGUMENTS]\n{tc.get('args', {})}",
                                "response": res.get("answer", "")
                            })
                        else:
                            # Fallback לכלים פשוטים
                            steps.append({
                                "module": f"EvidenceAnalyst/Tool:{tc.get('tool', 'unknown')}",
                                "prompt": f"[TOOL ARGUMENTS]\n{tc.get('args', {})}",
                                "response": str(res),
                            })
                    except StopIteration:
                        pass  # במקרה קיצון שאין התאמה בין הבקשות להיסטוריה

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
    # ── FIX: Force a final response if agent stopped on a tool call ───
    if not final_answer.strip():
        print("\n⚠️ Agent hit max iterations mid-thought (empty content). Forcing final text response...")

        # Filter out the last message if it's an unresolved tool call
        # to prevent OpenAI's "tool_calls must be followed by tool messages" 400 error.
        safe_messages = list(final_state["messages"])
        last_msg = safe_messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            safe_messages = safe_messages[:-1]  # Remove the hanging tool call

        # We use agent.llm (NOT agent.model) because agent.llm has NO tools bound.
        fallback_msg = agent.llm.invoke(
            safe_messages + [
                SystemMessage(content=(
                    "SYSTEM: You have reached the maximum allowed search queries. "
                    "You must now provide a final text answer to the user based ONLY on the "
                    "information you have gathered so far. Do not attempt to use any tools. "
                    "If you lack information, honestly state that."
                ))
            ]
        )
        final_answer = fallback_msg.content
        final_state["messages"].append(fallback_msg)
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