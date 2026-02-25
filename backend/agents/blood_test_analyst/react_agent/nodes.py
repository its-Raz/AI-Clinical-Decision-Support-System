"""
Node functions for Bloood test analyst ReAct agent.
"""

from langchain_core.messages import ToolMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ReActAgent

from state import ReActInternalState
import json
from utils import _print_messages,_print_response

def _format_tool_result(tool_name: str, result: any) -> str:
    """
    Format tool results into human-readable text instead of dict strings.

    Args:
        tool_name: Name of the tool
        result: Raw tool result

    Returns:
        Formatted string
    """
    if tool_name == "get_patient_history":
        # Format patient history nicely
        if isinstance(result, dict):
            formatted = f"""Patient: {result.get('name', 'Unknown')} ({result.get('age', '?')}yo {result.get('sex', '?')})

Chronic Conditions:
{', '.join(result.get('chronic_conditions', [])) or 'None'}

Lab History (Hemoglobin trend):
"""
            for entry in result.get('lab_history', []):
                hgb = entry.get('Hemoglobin', {})
                formatted += f"  {entry.get('date', '?')}: {hgb.get('value', '?')} {hgb.get('unit', '')} ({hgb.get('flag', 'unknown')})\n"

            formatted += "\nRecent Clinical Notes:\n"
            for note in result.get('recent_notes', []):
                formatted += f"  - {note}\n"

            return formatted.strip()

    elif tool_name == "check_reference_range":
        # Result is now a list — batch call covers all metrics at once
        if isinstance(result, list):
            parts = []
            for r in result:
                if "error" in r:
                    parts.append(f"  - {r.get('test_name', '?')}: {r['error']}")
                else:
                    parts.append(
                        f"  - {r.get('test_name','?')} = {r.get('value','?')} {r.get('unit','')} | "
                        f"Range: {r.get('reference_range','?')} | "
                        f"Status: {r.get('status','?')} ({r.get('severity','?')}) | "
                        f"Interpretation: {r.get('interpretation','?')}"
                    )
            return "Reference Range Results:\n" + "\n".join(parts)



    # Fallback: return as-is but try to make it cleaner
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


def call_tool(agent: 'ReActAgent', state: ReActInternalState) -> dict:
    """
    Tool node: Execute tools requested by the model.
    """
    outputs = []
    tool_history = []

    # Iterate over tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        print(f"🔧 Calling: {tool_call['name']}")

        # Get the tool and invoke it
        tool_result = agent.tools_by_name[tool_call["name"]].invoke(tool_call["args"])

        # --- Handle dictionary returned by RAG tool ---
        if tool_call["name"] == "search_medical_knowledge" and isinstance(tool_result, dict) and "answer" in tool_result:
            content_for_agent = f"Knowledge base answer:\n{tool_result['answer']}"
            history_result = tool_result
        else:
            # Handle other tools like history and reference range
            if tool_call["name"] in ["check_reference_range", "get_patient_history"]:
                tool_result = _format_tool_result(tool_call["name"], tool_result)
            content_for_agent = str(tool_result)
            history_result = str(tool_result)

        print(f"   ✅ Result: {content_for_agent[:80]}...")

        # Create ToolMessage with formatted result
        outputs.append(
            ToolMessage(
                content=content_for_agent,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

        # Record in history (keep raw result/dict)
        tool_history.append({
            "tool": tool_call["name"],
            "args": tool_call["args"],
            "result": history_result
        })

    # Combine with existing history
    updated_history = state.get("tool_calls_history", []) + tool_history

    return {
        "messages": outputs,
        "tool_calls_history": updated_history
    }


def call_model(agent: 'ReActAgent', state: ReActInternalState) -> dict:
    """
    Model node: Invoke the LLM.

    Args:
        agent: ReActAgent instance
        state: Current agent state

    Returns:
        Dict with model response
    """
    print(f"\n--- Iteration {state['iterations'] + 1}/{agent.max_iterations} ---")

    # Invoke the model with current messages

    _print_messages(state["messages"])
    response = agent.model.invoke(state["messages"])
    _print_response(response)
    # Return as list (will be added to messages by add_messages reducer)
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }