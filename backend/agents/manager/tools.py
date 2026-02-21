"""
backend/agents/manager/tools.py

Tools available to the Manager agent.
"""

from langchain_core.tools import tool
from typing import Literal


@tool
def judge_decision(
    accepted_category: Literal[
        "blood_test_analysis",
        "image_lesion_analysis",
        "evidence_analyst",
        "unsupported"
    ],
    reasoning: str,
    overridden: bool,
) -> dict:
    """
    Record the Judge's final routing decision.

    Call this tool to accept or override the semantic router's proposed category.

    Args:
        accepted_category: The final category you are committing to.
        reasoning: One sentence explaining why you accepted or overrode the proposal.
        overridden: True if you are changing the router's proposal, False if accepting it.
    """
    return {
        "accepted_category": accepted_category,
        "reasoning":         reasoning,
        "overridden":        overridden,
    }


__all__ = ["judge_decision"]