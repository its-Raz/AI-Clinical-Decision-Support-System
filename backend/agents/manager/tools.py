"""
backend/agents/manager/tools.py

Tools available to the Manager agent for routing and classification.
"""

from langchain_core.tools import tool
from typing import Literal


@tool
def classify_patient_request(user_message: str) -> dict:
    """
    Classify a patient's request to determine what service they need.

    Use this tool when a patient types a message describing what they want.
    Analyze the message to determine if they want:
    - Blood test analysis
    - Skin lesion analysis
    - Something else (unsupported)

    Args:
        user_message: The patient's message (e.g., "analyze my blood test")

    Returns:
        dict with:
        - intent: "blood_test" | "skin_lesion" | "unknown"
        - confidence: score 0-10
        - reasoning: why this classification was chosen

    Examples:
        "analyze my recent blood test" → blood_test
        "check my glucose levels" → blood_test
        "I have a mole I want checked" → skin_lesion
        "analyze this spot on my arm" → skin_lesion
        "schedule an appointment" → unknown
    """
    msg_lower = user_message.lower()

    # Blood test indicators
    blood_keywords = [
        "blood test", "lab result", "test result", "recent test",
        "lab work", "blood work", "analyze test", "check results",
        "glucose", "hemoglobin", "creatinine", "hba1c",
        "blood sugar", "lab values", "test values"
    ]

    # Skin lesion indicators
    lesion_keywords = [
        "lesion", "skin", "mole", "spot", "rash", "growth",
        "analyze image", "check skin", "dermatology", "picture",
        "bump", "mark", "freckle", "blemish", "discoloration"
    ]

    # Count matches
    blood_score = sum(2 if kw in msg_lower else 0 for kw in blood_keywords)
    lesion_score = sum(2 if kw in msg_lower else 0 for kw in lesion_keywords)

    # Classification logic
    if blood_score > lesion_score and blood_score >= 2:
        return {
            "intent": "blood_test",
            "confidence": min(blood_score, 10),
            "reasoning": f"Detected blood test keywords: matched {blood_score} indicators"
        }
    elif lesion_score > blood_score and lesion_score >= 2:
        return {
            "intent": "skin_lesion",
            "confidence": min(lesion_score, 10),
            "reasoning": f"Detected skin lesion keywords: matched {lesion_score} indicators"
        }
    else:
        return {
            "intent": "unknown",
            "confidence": 0,
            "reasoning": "No clear match to supported services (blood test or skin lesion analysis)"
        }


__all__ = ["classify_patient_request"]