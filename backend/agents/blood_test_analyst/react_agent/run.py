"""
Run function for the ReAct agent.

REFACTOR NOTE:
  lab_result may now be either a single dict (legacy) or a List[dict] (batch).
  All formatting and summary generation adapts to whichever form is received.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from __init__ import ReActAgent
from state import ReActInternalState

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompts import REACT_SYSTEM_PROMPT, REACT_PROMPT_TEMPLATE, SUMMARY_GENERATION_PROMPT


def _extract_react_steps(final_state: dict, initial_prompt: str) -> list[dict]:
    """
    Convert the ReAct agent's internal messages + tool_calls_history
    into structured step objects for the API trace in chronological order.
    """
    _TOOL_MODULE_NAMES = {
        "get_patient_history":      "Get Patient History Data",
        "check_reference_range":    "Check Reference Range",
        "search_medical_knowledge": "Search Medical Information",
    }

    steps = []
    messages = final_state.get("messages", [])
    tool_calls_history = final_state.get("tool_calls_history", [])
    history_iter = iter(tool_calls_history)

    from prompts import REACT_SYSTEM_PROMPT
    current_prompt = f"[SYSTEM]\n{REACT_SYSTEM_PROMPT}\n\n[USER]\n{initial_prompt}"

    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type == "ToolMessage":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            current_prompt = f"[TOOL OBSERVATION]\n{content}"

        elif msg_type == "AIMessage":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            response_text = content.strip()

            has_tools = hasattr(msg, "tool_calls") and msg.tool_calls

            if has_tools:
                tool_desc = ", ".join([f"{tc['name']}({tc['args']})" for tc in msg.tool_calls])
                response_text += f"\n[Tool Call Triggered: {tool_desc}]"

            if not response_text and has_tools:
                response_text = "[Tool Call Only]"

            if response_text:
                steps.append({
                    "module": "Blood Test Analyst",
                    "prompt": current_prompt,
                    "response": response_text.strip(),
                })

            if has_tools:
                for _ in msg.tool_calls:
                    try:
                        tc   = next(history_iter)
                        res  = tc.get("result", "")
                        tool = tc.get("tool", "unknown")
                        module_name = _TOOL_MODULE_NAMES.get(tool, tool)

                        if isinstance(res, dict) and "rag_sys_prompt" in res:
                            steps.append({
                                "module": "Search Medical Information",
                                "prompt": f"[TOOL ARGUMENTS]\n{tc.get('args', {})}",
                                "response": res.get("answer", "")
                            })
                        else:
                            steps.append({
                                "module": module_name,
                                "prompt": f"[TOOL ARGUMENTS]\n{tc.get('args', {})}",
                                "response": str(res),
                            })
                    except StopIteration:
                        pass

    return steps


def run_react_agent(global_state: dict) -> dict:
    print("\n" + "=" * 60)
    print("🔬 REACT AGENT: Starting Information Gathering")
    print("=" * 60)

    patient_id = global_state["patient_id"]
    lab_result = global_state["lab_result"]  # now either a dict or List[dict]

    # ── Normalise to list ─────────────────────────────────────────────────
    # Accepts both a single-metric dict (legacy) and a list (batch refactor).
    if isinstance(lab_result, dict):
        lab_results = [lab_result]
    else:
        lab_results = lab_result   # already a list

    # ── Format the input block for the prompt ─────────────────────────────
    if len(lab_results) == 1:
        r = lab_results[0]
        input_text = (
            f"Lab Result: Patient {patient_id} — "
            f"{r['test_name']} {r['value']} {r['unit']} "
            f"(Flag: {r.get('flag', 'N/A')})\n"
        )
        print(f"   Patient : {patient_id}")
        print(f"   Lab     : {r['test_name']} = {r['value']} {r['unit']}")
    else:
        lines = "\n".join(
            f"  - {r['test_name']}: {r['value']} {r['unit']} (Flag: {r.get('flag', 'N/A')})"
            for r in lab_results
        )
        input_text = (
            f"Lab Results: Patient {patient_id} — {len(lab_results)} abnormal metrics:\n"
            f"{lines}\n"
        )
        print(f"   Patient : {patient_id}")
        print(f"   Metrics : {[r['test_name'] for r in lab_results]}")

    user_prompt = REACT_PROMPT_TEMPLATE.format(input=input_text)

    inputs: ReActInternalState = {
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        "iterations":         0,
        "tool_calls_history": [],
    }

    agent = ReActAgent()

    print("\n🤖 Running ReAct loop...\n")

    final_state = None
    for state in agent.graph.stream(inputs, stream_mode="values"):
        final_state = state

    observations       = _extract_observations(final_state)
    tool_calls_history = final_state.get("tool_calls_history", [])

    print(f"\n📊 ReAct Phase Complete:")
    print(f"   Tools used : {list(observations.keys())}")
    print(f"   Tool calls : {len(tool_calls_history)}")
    print(f"   Iterations : {final_state['iterations']}")

    print("\n📝 Generating consolidated summary from observations...")
    summary, summary_prompt = _generate_summary(agent, observations, patient_id, lab_results)
    print(f"   Summary length: {len(summary)} chars")

    agent_steps = _extract_react_steps(final_state, user_prompt)
    agent_steps.append({
        "module":   "Final Analysis",
        "prompt":   f"[USER]\n{summary_prompt}",
        "response": summary,
    })

    return {
        "lab_insights": summary,
        "steps":        agent_steps,
    }


def _extract_observations(final_state: ReActInternalState) -> dict:
    """Collect all tool outputs keyed by tool name."""
    observations  = {}
    tool_call_map = {}

    for message in final_state["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_map[tool_call["id"]] = tool_call["name"]

    for message in final_state["messages"]:
        if isinstance(message, ToolMessage):
            tool_name = tool_call_map.get(message.tool_call_id, "unknown")
            if tool_name not in observations:
                observations[tool_name] = []
            observations[tool_name].append(message.content)

    return observations


def _generate_summary(
    agent,
    observations: dict,
    patient_id: str,
    lab_results: list,
) -> tuple[str, str]:
    """
    Build a consolidated summary prompt covering ALL lab results and invoke
    a dedicated lightweight LLM instance to produce a single clinical narrative.
    """
    from langchain_openai import ChatOpenAI
    import os

    # ── Dedicated summary LLM — built from config, no reasoning overhead ──
    llm_config = agent.config['llm']
    api_key    = os.getenv(llm_config['api_key_env'])

    summary_llm = ChatOpenAI(
        model=llm_config['model'],
        temperature=1.0,
        openai_api_key=api_key,
        base_url=llm_config.get('base_url'),
        max_tokens=2000,
        reasoning_effort="low"
    )

    # ── Patient history ───────────────────────────────────────────────────
    patient_history = ""
    if "get_patient_history" in observations:
        patient_history = f"1. PATIENT HISTORY:\n{observations['get_patient_history'][0]}"

    # ── Reference range checks ────────────────────────────────────────────
    reference_checks = ""
    if "check_reference_range" in observations:
        entries = observations["check_reference_range"]
        if len(entries) == 1:
            reference_checks = f"\n2. REFERENCE RANGE CHECK:\n{entries[0]}"
        else:
            numbered = "\n\n".join(f"  [{i+1}] {e}" for i, e in enumerate(entries))
            reference_checks = f"\n2. REFERENCE RANGE CHECKS ({len(entries)} metrics):\n{numbered}"

    # ── Medical knowledge ─────────────────────────────────────────────────
    medical_knowledge = ""
    if "search_medical_knowledge" in observations:
        entries = observations["search_medical_knowledge"]
        if len(entries) == 1:
            medical_knowledge = f"\n3. MEDICAL KNOWLEDGE:\n{entries[0]}"
        else:
            numbered = "\n\n".join(f"  [{i+1}] {e}" for i, e in enumerate(entries))
            medical_knowledge = f"\n3. MEDICAL KNOWLEDGE ({len(entries)} searches):\n{numbered}"

    # ── Lab results header ────────────────────────────────────────────────
    if len(lab_results) == 1:
        r = lab_results[0]
        lab_results_summary = (
            f"{r['test_name']} = {r['value']} {r['unit']} (Flag: {r.get('flag', 'N/A')})"
        )
    else:
        lines = "\n".join(
            f"  - {r['test_name']}: {r['value']} {r['unit']} (Flag: {r.get('flag', 'N/A')})"
            for r in lab_results
        )
        lab_results_summary = f"{len(lab_results)} abnormal metrics:\n{lines}"

    summary_prompt = SUMMARY_GENERATION_PROMPT.format(
        patient_id          = patient_id,
        lab_results_summary = lab_results_summary,
        patient_history     = patient_history,
        reference_checks    = reference_checks,
        medical_knowledge   = medical_knowledge,
    )

    print(f"📝 [_generate_summary] Invoking summary LLM (model={llm_config['model']}, max_tokens=2500, temp=1)...")
    response = summary_llm.invoke([HumanMessage(content=summary_prompt)])

    # ── Debug token usage ─────────────────────────────────────────────────
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    reasoning_tokens  = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    print(f"   tokens — prompt: {usage.get('prompt_tokens', 0)} | completion: {completion_tokens} | reasoning: {reasoning_tokens}")

    if not response.content:
        print("⚠️  [_generate_summary] Empty response — returning fallback")
        return "Summary generation failed. Please review the raw observations above.", summary_prompt

    return response.content, summary_prompt