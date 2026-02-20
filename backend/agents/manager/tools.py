"""
backend/agents/manager/tools.py

Tools available to the Manager agent for routing and classification.
"""

from langchain_core.tools import tool
from typing import Literal


@tool
def classify_patient_request(user_message: str) -> dict:



__all__ = ["classify_patient_request"]