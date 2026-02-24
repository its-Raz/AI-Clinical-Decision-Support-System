"""
Run function for the ReAct agent.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from __init__ import ReActAgent
from state import ReActInternalState

# Import prompts
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompts import REACT_SYSTEM_PROMPT, REACT_PROMPT_TEMPLATE, SUMMARY_GENERATION_PROMPT


def _extract_react_steps(final_state: dict, initial_prompt: str) -> list[dict]:
    """
    Convert the ReAct agent's internal messages + tool_calls_history
    into structured step objects for the API trace in chronological order.
    """
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
                    "module": "BloodTestAnalyst/ReAct",
                    "prompt": current_prompt,
                    "response": response_text.strip(),
                })

            if has_tools:
                for _ in msg.tool_calls:
                    try:
                        tc = next(history_iter)
                        res = tc.get("result", "")

                        if isinstance(res, dict) and "rag_sys_prompt" in res:
                            steps.append({
                                "module": "BloodTestAnalyst/RAG_LLM",
                                "prompt": f"[SYSTEM]\n{res['rag_sys_prompt']}\n\n[USER]\n{res['rag_user_prompt']}",
                                "response": res.get("answer", "")
                            })
                            steps.append({
                                "module": f"BloodTestAnalyst/Tool:{tc.get('tool', 'unknown')}",
                                "prompt": f"[TOOL ARGUMENTS]\n{tc.get('args', {})}",
                                "response": res.get("answer", "")
                            })
                        else:
                            steps.append({
                                "module": f"BloodTestAnalyst/Tool:{tc.get('tool', 'unknown')}",
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

    patient_id = global_state['patient_id']
    lab_result = global_state['lab_result']

    print(f"Patient: {patient_id}")
    print(f"Lab: {lab_result['test_name']} = {lab_result['value']} {lab_result['unit']}")

    input_text = f"Lab Result: Patient {patient_id} -  {lab_result['test_name']} {lab_result['value']} {lab_result['unit']} ({lab_result.get('flag', 'N/A')})\n"
    user_prompt = REACT_PROMPT_TEMPLATE.format(input=input_text)

    inputs: ReActInternalState = {
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ],
        "iterations": 0,
        "tool_calls_history": []
    }

    agent = ReActAgent()

    print("\n🤖 Running ReAct loop...\n")

    final_state = None
    for state in agent.graph.stream(inputs, stream_mode="values"):
        final_state = state

    observations = _extract_observations(final_state)
    tool_calls_history = final_state.get("tool_calls_history", [])

    print(f"\n📊 ReAct Phase Complete:")
    print(f"   Tools used: {len(observations)}")
    print(f"   Tool calls: {len(tool_calls_history)}")
    print(f"   Iterations: {final_state['iterations']}")

    print("\n📝 Generating summary from observations...")
    # NOTE: We now unpack two values to get the prompt for the trace
    summary, summary_prompt = _generate_summary(agent, observations, patient_id, lab_result)
    print(f"   Summary length: {len(summary)} chars")

    # Build chronological trace
    agent_steps = _extract_react_steps(final_state, user_prompt)

    # Append the Summary Generation step to the end of the trace!
    agent_steps.append({
        "module": "BloodTestAnalyst/SummaryGenerator",
        "prompt": f"[USER]\n{summary_prompt}",
        "response": summary
    })

    # Return only the fields we want to update in the global graph state
    return {
        "lab_insights": summary,
        "steps": agent_steps
    }


def _extract_observations(final_state: ReActInternalState) -> dict:
    observations = {}
    tool_call_map = {}
    for message in final_state["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_map[tool_call['id']] = tool_call['name']

    for message in final_state["messages"]:
        if isinstance(message, ToolMessage):
            tool_name = tool_call_map.get(message.tool_call_id, "unknown")
            if tool_name not in observations:
                observations[tool_name] = []
            observations[tool_name].append(message.content)

    return observations


def _generate_summary(agent, observations: dict, patient_id: str, lab_result: dict) -> tuple[str, str]:
    patient_history = ""
    if 'get_patient_history' in observations:
        patient_history = f"1. PATIENT HISTORY:\n{observations['get_patient_history'][0]}"

    reference_check = ""
    if 'check_reference_range' in observations:
        reference_check = f"\n2. REFERENCE RANGE CHECK:\n{observations['check_reference_range'][0]}"

    medical_knowledge = ""
    if 'search_medical_knowledge' in observations:
        medical_knowledge = f"\n3. MEDICAL KNOWLEDGE:\n{observations['search_medical_knowledge'][0]}"

    summary_prompt = SUMMARY_GENERATION_PROMPT.format(
        patient_id=patient_id,
        test_name=lab_result['test_name'],
        value=lab_result['value'],
        unit=lab_result['unit'],
        flag=lab_result.get('flag', 'N/A'),
        patient_history=patient_history,
        reference_check=reference_check,
        medical_knowledge=medical_knowledge
    )

    response = agent.llm.invoke([HumanMessage(content=summary_prompt)])

    # Return both the generated text AND the prompt we used
    return response.content, summary_prompt