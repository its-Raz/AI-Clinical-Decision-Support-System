"""
ReAct Node: Information Gathering Phase
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import yaml
import os

# Load environment variables from .env
load_dotenv()

# Import tools
from ..tools import get_patient_history, check_reference_range, search_medical_knowledge

# Import prompts
from ..prompts import REACT_SYSTEM_PROMPT, REACT_PROMPT_TEMPLATE

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)


def react_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct node: Gather information using tools.

    Args:
        state: ClinicalState dict containing:
            - patient_id: str
            - patient_info: dict (age, sex)
            - lab_result: dict (test_name, value, unit, etc.)

    Returns:
        Dict with updates:
            - observations: Results from tools
            - tool_calls: List of tool calls made
            - react_iterations: Number of iterations used
            - react_summary: Agent's final analysis
    """

    print("\n" + "="*60)
    print("🔬 REACT NODE: Starting Information Gathering")
    print("="*60)

    # Extract state
    patient_id = state.get("patient_id")
    patient_info = state.get("patient_info", {})
    lab_result = state.get("lab_result", {})

    print(f"Patient: {patient_id}")
    print(f"Lab: {lab_result.get('test_name')} = {lab_result.get('value')} {lab_result.get('unit')}")

    # Get API key from environment
    api_key = os.getenv(CONFIG['llm']['api_key_env'])
    if not api_key:
        raise ValueError(f"Missing {CONFIG['llm']['api_key_env']} in .env file")

    # Initialize LLM
    llm = ChatOpenAI(
        model=CONFIG['llm']['model'],
        base_url=CONFIG['llm']['base_url'],
        openai_api_key=api_key, # Load from .env
        max_tokens=CONFIG['llm']['max_tokens'],temperature = 1,

    )

    # Tools
    tools = [get_patient_history, check_reference_range, search_medical_knowledge]

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", REACT_SYSTEM_PROMPT),
        ("human", REACT_PROMPT_TEMPLATE),
        MessagesPlaceholder("agent_scratchpad")
    ])

    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=CONFIG['react']['max_iterations'],
        verbose=CONFIG['react']['verbose'],
        return_intermediate_steps=True
    )

    # Build input
    user_input = f"""Patient: {patient_id} ({patient_info.get('age')}yo {patient_info.get('sex')})
Lab: {lab_result.get('test_name')} = {lab_result.get('value')} {lab_result.get('unit')} (Flag: {lab_result.get('flag', 'N/A')})

Gather patient history, check reference ranges, and search medical causes."""
    print(f'---- USER INPUT ----')
    print(user_input)
    # Execute
    try:
        print("\n🤖 Running ReAct Agent...\n")
        result = agent_executor.invoke({"input": user_input})

        # Extract data
        observations = {}
        tool_calls = []

        if "intermediate_steps" in result:
            for action, obs in result["intermediate_steps"]:
                tool_name = action.tool

                # Store observation
                if tool_name not in observations:
                    observations[tool_name] = []
                observations[tool_name].append(obs)

                # Record tool call
                tool_calls.append({
                    "tool": tool_name,
                    "input": action.tool_input,
                    "output": obs
                })

                print(f"✅ Tool: {tool_name}")
                print(f"   Input: {action.tool_input}")
                print(f"   Output: {str(obs)[:100]}...")

        agent_summary = result.get("output", "")

        print(f"\n📊 ReAct Summary:")
        print(f"   Tools used: {len(tool_calls)}")
        print(f"   Iterations: {len(result.get('intermediate_steps', []))}")
        print(f"   Summary: {agent_summary}...")
        print(f'Agent Results:')
        print(result)
        print(f'Agent Observations: ')
        print(observations)
        return {
            "observations": observations,
            "tool_calls": tool_calls,
            "react_iterations": len(result.get("intermediate_steps", [])),
            "react_summary": agent_summary
        }

    except Exception as e:
        print(f"\n❌ ReAct Error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "observations": {},
            "tool_calls": [],
            "react_iterations": 0,
            "react_error": str(e)
        }