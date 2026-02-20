"""
Edge functions for routing in the graph.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ReActAgent

from backend.agents.evidence_analyst.state import ReActInternalState


def should_continue(agent: 'ReActAgent', state: ReActInternalState) -> str:
    """
    Determine whether to continue or end.

    Args:
        agent: ReActAgent instance
        state: Current agent state

    Returns:
        "continue" to call tools, "end" to finish
    """
    messages = state["messages"]

    # Check max iterations
    if state["iterations"] >= agent.max_iterations:
        print(f"⚠️  Max iterations ({agent.max_iterations}) reached")
        return "end"

    # If the last message has tool calls, continue
    if messages[-1].tool_calls:
        return "continue"

    # Otherwise, end
    print("✅ Agent finished")
    return "end"