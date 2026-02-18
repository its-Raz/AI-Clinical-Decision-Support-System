"""
backend/graph.py — System entry point.

Provides:
  build_system()            → ManagerAgent (cached singleton)
  trigger_p002_lab_result() → initial AgentState for the demo button

P002 lab batch is pulled directly from the existing patient mockup data.
"""

import operator
from functools import lru_cache
from typing import Annotated, List, Optional, TypedDict

# ── AgentState (re-exported so callers only import from here) ─────────────
from backend.agents.global_state import AgentState

# ── Patient mockup ────────────────────────────────────────────────────────
from backend.agents.blood_test_analyst.data_mockups.patients_mockup import get_patient


# ─────────────────────────────────────────────────────────────────────────
# System builder
# ─────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_system():
    """Return a cached ManagerAgent instance (loads models once)."""
    from backend.agents.manager import ManagerAgent
    return ManagerAgent()


# ─────────────────────────────────────────────────────────────────────────
# Demo trigger — P002 blood test batch
# ─────────────────────────────────────────────────────────────────────────

def trigger_p002_lab_result() -> AgentState:
    """
    Simulate a new lab-result event for patient P002.

    Pulls the most recent lab entry from the patient mockup and converts
    it into the flat list-of-dicts format expected by AgentState.
    """
    patient   = get_patient("P002")
    lab_history = patient.get("lab_history", [])

    if not lab_history:
        raise ValueError("No lab history found for P002 in mockup data.")

    # Most recent batch = last entry in history
    latest_batch = lab_history[-1]

    print(f"\n🔬 [trigger_p002_lab_result] Loading P002 data...")
    print(f"   Latest batch date: {latest_batch.get('date', 'N/A')}")
    print(f"   Raw batch keys: {list(latest_batch.keys())}")

    # Convert {TestName: {value, unit, flag}} → List[{test_name, value, unit, flag}]
    lab_results = []
    for test_name, data in latest_batch.items():
        if test_name == "date":
            continue
        lab_results.append({
            "test_name": test_name,
            "value":     data.get("value"),
            "unit":      data.get("unit", ""),
            "flag":      data.get("flag", "normal"),
        })

    print(f"   Converted to {len(lab_results)} lab_result entries")
    for r in lab_results:
        print(f"      • {r['test_name']:15} = {r['value']} {r['unit']:8} (flag={r['flag']})")

    state: AgentState = {
        "request_type":   "blood_test_analysis",
        "patient_id":     "P002",
        "lab_result":     lab_results,
        "lab_insights":   None,
        "image_path":     None,
        "vision_results": None,
        "vision_insights": None,
        "messages":       [],
        "next_step":      "",
        "final_report":   None,
    }

    return state