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


def run_react_agent(global_state: dict) -> dict:
    """
    Run the ReAct agent on a clinical case.

    Creates a fresh ReActAgent instance and runs it.

    Args:
        global_state: Dict with:
            - patient_id: str
            - patient_info: dict (age, sex)
            - lab_result: dict (test_name, value, unit, flag)

    Returns:
        Updated global_state with:
            - observations: dict
            - react_summary: str
            - tool_calls_history: list
    """
    print("\n" + "="*60)
    print("🔬 REACT AGENT: Starting Information Gathering")
    print("="*60)

    # Extract patient info
    patient_id = global_state['patient_id']
    patient_info = global_state['patient_info']
    lab_result = global_state['lab_result']

    print(f"Patient: {patient_id}")
    print(f"Lab: {lab_result['test_name']} = {lab_result['value']} {lab_result['unit']}")

    # Build input for the template
    input_text = f"""Lab Result: Patient {patient_id} -  {lab_result['test_name']} {lab_result['value']} {lab_result['unit']} ({lab_result.get('flag', 'N/A')})
"""

    # Format the prompt template with the input
    user_prompt = REACT_PROMPT_TEMPLATE.format(input=input_text)

    # Create initial state
    inputs: ReActInternalState = {
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ],
        "iterations": 0,
        "tool_calls_history": []
    }

    # Create fresh agent instance
    agent = ReActAgent()

    # Run the graph with streaming
    print("\n🤖 Running ReAct loop...\n")

    final_state = None
    for state in agent.graph.stream(inputs, stream_mode="values"):
        final_state = state

    # Extract results
    observations = _extract_observations(final_state)
    tool_calls_history = final_state.get("tool_calls_history", [])

    print(f"\n📊 ReAct Phase Complete:")
    print(f"   Tools used: {len(observations)}")
    print(f"   Tool calls: {len(tool_calls_history)}")
    print(f"   Iterations: {final_state['iterations']}")

    # Generate summary
    print("\n📝 Generating summary from observations...")
    summary = _generate_summary(agent, observations, patient_id, patient_info, lab_result)
    print(f"   Summary length: {len(summary)} chars")

    # Update global state
    global_state['observations'] = observations
    global_state['react_summary'] = summary
    global_state['tool_calls_history'] = tool_calls_history

    return global_state


def _extract_observations(final_state: ReActInternalState) -> dict:
    """Extract tool results from message history."""
    observations = {}

    # Build mapping of tool_call_id to tool_name
    tool_call_map = {}
    for message in final_state["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_map[tool_call['id']] = tool_call['name']

    # Extract tool results
    for message in final_state["messages"]:
        if isinstance(message, ToolMessage):
            tool_name = tool_call_map.get(message.tool_call_id, "unknown")

            if tool_name not in observations:
                observations[tool_name] = []

            observations[tool_name].append(message.content)

    return observations


def _generate_summary(agent, observations: dict, patient_id: str, patient_info: dict, lab_result: dict) -> str:
    """
    Generate a comprehensive summary from all observations.

    This is a separate LLM call after the ReAct loop completes.

    Args:
        agent: ReActAgent instance (for accessing llm)
        observations: Tool results collected during ReAct
        patient_id: Patient ID
        patient_info: Patient demographics
        lab_result: Lab test result

    Returns:
        Comprehensive summary string
    """

    # Format each observation section
    patient_history = ""
    if 'get_patient_history' in observations:
        patient_history = f"1. PATIENT HISTORY:\n{observations['get_patient_history'][0]}"

    reference_check = ""
    if 'check_reference_range' in observations:
        reference_check = f"\n2. REFERENCE RANGE CHECK:\n{observations['check_reference_range'][0]}"

    medical_knowledge = ""
    if 'search_medical_knowledge' in observations:
        medical_knowledge = f"\n3. MEDICAL KNOWLEDGE:\n{observations['search_medical_knowledge'][0]}"

    # Format the summary prompt using the template
    summary_prompt = SUMMARY_GENERATION_PROMPT.format(
        patient_id=patient_id,
        age=patient_info['age'],
        sex=patient_info['sex'],
        test_name=lab_result['test_name'],
        value=lab_result['value'],
        unit=lab_result['unit'],
        flag=lab_result.get('flag', 'N/A'),
        patient_history=patient_history,
        reference_check=reference_check,
        medical_knowledge=medical_knowledge
    )

    # Call LLM for summary (without tools)
    response = agent.llm.invoke([HumanMessage(content=summary_prompt)])

    return response.content