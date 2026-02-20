"""
backend/agents/manager/tools.py

Tools available to the Manager agent for routing and classification.
"""

from langchain_core.tools import tool
from typing import Literal

@tool
def classify_patient_request(
    category: Literal["blood_test_analysis", "image_lesion_analysis", "evidence_analyst", "unsupported"],
    reasoning: str
) -> dict:
    """
    Classify the incoming patient request into one of the supported medical categories.

    Categories:
    - blood_test_analysis: The user is asking about lab results or blood tests.
    - image_lesion_analysis: The user uploaded or is asking about a skin image/lesion/mole.
    - evidence_analyst: The user is asking a general medical question, about symptoms, or treatments.
    - unsupported: The request is NOT medical, or it is a medical request we cannot safely handle.

    Args:
        category: The exact category string.
        reasoning: A short explanation of why this category was chosen.
    """
    return {"category": category, "reasoning": reasoning}

__all__ = ["classify_patient_request"]