"""
Node functions for ReAct agent.
"""

from langchain_core.messages import ToolMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ReActAgent

from backend.agents.evidence_analyst.state import ReActInternalState
import json
from backend.agents.evidence_analyst.utils import _print_messages,_print_response



def call_tool(agent: 'ReActAgent', state: ReActInternalState) -> dict:
    """
    Tool node: Execute tools requested by the model.

    Args:
        agent: ReActAgent instance
        state: Current agent state

    Returns:
        Dict with tool results as messages and updated tool_calls_history
    """
    outputs = []
    tool_history = []

    # Iterate over tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        print(f"🔧 Calling: {tool_call['name']}")

        # Get the tool and invoke it
        tool_result = agent.tools_by_name[tool_call["name"]].invoke(tool_call["args"])

        print(f"   ✅ Result: {str(tool_result)[:80]}...")


        # Create ToolMessage with formatted result
        outputs.append(
            ToolMessage(
                content=tool_result,  # ← Use formatted version
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

        # Record in history (keep raw result)
        tool_history.append({
            "tool": tool_call["name"],
            "args": tool_call["args"],
            "result": str(tool_result)  # Keep original for debugging
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