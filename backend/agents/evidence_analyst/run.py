"""
Run function for the ReAct agent.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from backend.agents.evidence_analyst.__init__ import ReActAgent
from backend.agents.evidence_analyst.state import ReActInternalState

# Import prompts
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.agents.evidence_analyst.prompts import REACT_SYSTEM_PROMPT
from backend.agents.evidence_analyst.utils import extract_latest_user_query

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

    query = extract_latest_user_query(global_state)

    print(f"Patient: {patient_id}")
    print(f'Query: {query}')




    # Create initial state
    inputs: ReActInternalState = {
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=query)
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

    final_answer = final_state["messages"][-1].content
    global_state["evidence_insights"] = final_answer
    return global_state





