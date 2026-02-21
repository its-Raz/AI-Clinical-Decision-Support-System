"""
backend/agents/global_state.py
Shared AgentState schema used by all agents and the Manager graph.
"""

import operator
from typing import Annotated, List, Optional, TypedDict


class AgentState(TypedDict):
    # ── Classification ─────────────────────────────────────────────────
    request_type:   str                  # final accepted category (written by Judge)
    patient_id:     str

    # ── Semantic router metadata ────────────────────────────────────────
    raw_user_input:              Optional[str]    # original free-text from user
    router_proposed_category:    Optional[str]    # what the router suggested
    router_score:                Optional[float]  # cosine similarity score
    router_confidence:           Optional[str]    # "high" | "medium" | "spam"

    # ── Specialist payloads ─────────────────────────────────────────────
    lab_result:      Optional[List[dict]]   # [{test_name, value, unit, flag}, ...]
    lab_insights:    Optional[str]          # blood test analyst summary
    image_path:      Optional[str]
    vision_results:  Optional[dict]         # {bbox, label, conf} - raw YOLO output
    vision_insights: Optional[str]          # skin care analyst clinical summary
    evidence_insights: Optional[str]

    # ── Graph internals ─────────────────────────────────────────────────
    messages:        Annotated[List[dict], operator.add]
    next_step:       str
    final_report:    Optional[str]          # patient-facing delivery message (reshaped by manager)

