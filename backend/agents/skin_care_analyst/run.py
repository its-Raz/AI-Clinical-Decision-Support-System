"""
Entry point for the Skin Care Classifier Agent.
Called by the Agent Manager when request_type == "image_lesion_analysis".
"""

from ..global_state import AgentState
from .agent import SkinCareAgent


def run_skin_care_agent(state: AgentState) -> AgentState:
    """
    Run the Skin Care Classifier Agent.

    Args:
        state: Shared AgentState. Must have `image_path` and `patient_id` set.

    Returns:
        Updated state with `vision_results` and `final_report` populated.
    """
    if not state.get("image_path"):
        raise ValueError("AgentState must contain `image_path` to run the Skin Care Agent.")

    agent = SkinCareAgent()
    return agent.run(state)
