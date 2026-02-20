"""
backend/agents/global_state.py
Shared AgentState schema used by all agents and the Manager graph.
"""

import operator
from typing import Annotated, List, Optional, TypedDict


class AgentState(TypedDict):
    request_type:   str                          # "blood_test_analysis" | "image_lesion_analysis"
    patient_id:     str
    lab_result:     Optional[List[dict]]         # [{test_name, value, unit, flag}, ...]
    lab_insights:   Optional[str]                # blood test analyst summary
    image_path:     Optional[str]
    vision_results: Optional[dict]               # {bbox, label, conf} - raw YOLO output
    vision_insights: Optional[str]               # skin care analyst clinical summary
    evidence_insights: Optional[str]
    messages:       Annotated[List[dict], operator.add]   # all trace + patient messages
    next_step:      str
    final_report:   Optional[str]                # patient-facing delivery message (reshaped by manager)

