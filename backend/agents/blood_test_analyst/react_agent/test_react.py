"""
Test the ReAct Agent.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import run_react_agent


def test_react_agent():
    """Test ReAct agent with patient P001."""

    # Create global state
    global_state = {
        "patient_id": "P001",

        "lab_result": {
            "test_name": "Hemoglobin",
            "value": 9.8,
            "unit": "g/dL",
            "flag": "low"
        }
    }

    # Run agent
    result = run_react_agent(global_state)

    # Check results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Observations: {list(result['observations'].keys())}")
    print(f"Tool calls history: {len(result['tool_calls_history'])} calls")

    # Print tool calls history
    print("\nTool Calls History:")
    for i, call in enumerate(result['tool_calls_history'], 1):
        print(f"  {i}. {call['tool']} -> {call['result'][:60]}...")

    print(f"\nSummary:\n{result['react_summary']}...")

    assert len(result['observations']) > 0, "Should have observations"
    assert len(result['react_summary']) > 0, "Should have summary"
    assert len(result['tool_calls_history']) > 0, "Should have tool calls history"

    print("\n✅ Test passed!")


if __name__ == "__main__":
    test_react_agent()