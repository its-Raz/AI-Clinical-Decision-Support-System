"""
backend/graph.py — System entry point with Supabase integration.

Provides:
  build_system()              → ManagerAgent (cache disabled for debugging)
  simulate_new_lab_result()   → Adds NEW lab result to Supabase, triggers analysis
  analyze_existing_test()     → Analyzes existing test from Supabase (NEW)

REALISTIC WORKFLOWS:
  1. Generate new test: lab arrives → DB stores → analyze
  2. Review existing test: browse patients → select test → analyze
"""

from datetime import datetime
import random

# ── AgentState (re-exported so callers only import from here) ─────────────
from backend.agents.global_state import AgentState


# ─────────────────────────────────────────────────────────────────────────
# System builder
# ─────────────────────────────────────────────────────────────────────────

# TEMPORARY: Cache disabled to force reload after code changes
# Re-enable after confirming the fix works
# @lru_cache(maxsize=1)
def build_system():
    """Return a ManagerAgent instance (cache temporarily disabled for debugging)."""
    from backend.agents.manager import ManagerAgent
    print("🔄 [build_system] Creating NEW ManagerAgent instance (cache disabled)")
    return ManagerAgent()


# ─────────────────────────────────────────────────────────────────────────
# Demo trigger — NEW LAB RESULT workflow
# ─────────────────────────────────────────────────────────────────────────

def simulate_new_lab_result() -> AgentState:
    """
    REALISTIC WORKFLOW: Simulate a new lab result arriving from the lab system.

    Steps:
    1. Fetch the first patient from Supabase
    2. Generate a new lab result entry
    3. Add it to the patient's lab_history in Supabase
    4. Extract the new result for analysis
    5. Return AgentState for the manager to process

    This mimics production: lab results arrive → stored in DB → trigger analysis.
    """
    from backend.supabase.supabase_client import get_first_patient, add_lab_result

    print("\n" + "=" * 70)
    print("🔬 SIMULATING NEW LAB RESULT ARRIVAL")
    print("=" * 70)

    # ── Step 1: Fetch patient from Supabase ──────────────────────────────
    patient = get_first_patient()

    if not patient:
        raise ValueError("No patients found in Supabase database.")

    patient_id = patient["id"]
    patient_name = patient["name"]
    existing_history = patient.get("lab_history", [])

    print(f"\n📋 Patient: {patient_id} ({patient_name})")
    print(f"   Existing lab history: {len(existing_history)} entries")

    # ── Step 2: Generate new lab result ──────────────────────────────────
    # Simulate realistic values with some being abnormal
    today = datetime.now().strftime("%Y-%m-%d")

    # Get last values if available to create trends
    if existing_history:
        last_entry = existing_history[-1]
        last_glucose = last_entry.get("Glucose", {}).get("value", 95)
        last_hgb = last_entry.get("Hemoglobin", {}).get("value", 14.0)
        last_cre = last_entry.get("Creatinine", {}).get("value", 1.0)
    else:
        last_glucose = 95
        last_hgb = 14.0
        last_cre = 1.0

    # Add variation (some patients trending up in glucose)
    glucose_value = int(last_glucose + random.randint(-3, 8))
    hgb_value = round(last_hgb + random.uniform(-0.5, 0.5), 1)
    cre_value = round(last_cre + random.uniform(-0.1, 0.1), 2)

    # Determine flags
    glucose_flag = "high" if glucose_value > 100 else "normal"

    if patient["sex"] == "M":
        hgb_flag = "low" if hgb_value < 13.5 else "normal"
    else:
        hgb_flag = "low" if hgb_value < 12.0 else "normal"

    cre_flag = "high" if cre_value > 1.3 else "normal"

    new_lab_entry = {
        "date": today,
        "Hemoglobin": {"value": hgb_value, "unit": "g/dL", "flag": hgb_flag},
        "Glucose": {"value": glucose_value, "unit": "mg/dL", "flag": glucose_flag},
        "Creatinine": {"value": cre_value, "unit": "mg/dL", "flag": cre_flag},
    }

    print(f"\n🧪 NEW LAB RESULT generated for {today}:")
    print(f"   Glucose:     {glucose_value} mg/dL  [{glucose_flag}]")
    print(f"   Hemoglobin:  {hgb_value} g/dL   [{hgb_flag}]")
    print(f"   Creatinine:  {cre_value} mg/dL  [{cre_flag}]")

    # ── Step 3: Add to Supabase ──────────────────────────────────────────
    print(f"\n💾 Adding new lab result to Supabase...")
    success = add_lab_result(patient_id, new_lab_entry)

    if not success:
        raise RuntimeError(f"Failed to add lab result to Supabase for {patient_id}")

    print(f"   ✅ Lab result stored in database")

    # ── Step 4: Extract for analysis ─────────────────────────────────────
    # Convert to the format the agent expects
    lab_results = []
    for test_name, data in new_lab_entry.items():
        if test_name == "date":
            continue
        lab_results.append({
            "test_name": test_name,
            "value":     data["value"],
            "unit":      data["unit"],
            "flag":      data["flag"],
        })

    print(f"\n🎯 Triggering analysis workflow...")
    print(f"   Metrics to analyze: {len(lab_results)}")
    print("=" * 70 + "\n")

    # ── Step 5: Build AgentState ─────────────────────────────────────────
    state: AgentState = {
        "request_type":   "blood_test_analysis",
        "patient_id":     patient_id,
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


# ─────────────────────────────────────────────────────────────────────────
# Demo trigger — EXISTING TEST workflow (NEW)
# ─────────────────────────────────────────────────────────────────────────

def analyze_existing_test(patient_id: str, test_index: int) -> AgentState:
    """
    Trigger analysis on an existing lab test from Supabase.

    Args:
        patient_id: Patient ID like "P001"
        test_index: Index of the test in lab_history (0 = oldest, -1 = most recent)

    Returns:
        AgentState ready for analysis
    """
    from backend.supabase.supabase_client import fetch_patient_by_id

    print("\n" + "=" * 70)
    print("📋 ANALYZING EXISTING LAB TEST")
    print("=" * 70)

    # Fetch patient
    patient = fetch_patient_by_id(patient_id)

    if not patient:
        raise ValueError(f"Patient {patient_id} not found in Supabase.")

    patient_name = patient["name"]
    lab_history = patient.get("lab_history", [])

    if not lab_history:
        raise ValueError(f"No lab history found for patient {patient_id}.")

    # Get selected test
    if test_index < 0:
        test_index = len(lab_history) + test_index  # Convert negative index

    if test_index < 0 or test_index >= len(lab_history):
        raise ValueError(f"Invalid test index {test_index}. Patient has {len(lab_history)} tests.")

    selected_test = lab_history[test_index]
    test_date = selected_test.get("date", "Unknown")

    print(f"\n📋 Patient: {patient_id} ({patient_name})")
    print(f"   Total lab history: {len(lab_history)} entries")
    print(f"   Selected test: #{test_index + 1} (Date: {test_date})")

    # Convert to agent format
    lab_results = []
    for test_name, data in selected_test.items():
        if test_name == "date":
            continue
        lab_results.append({
            "test_name": test_name,
            "value":     data["value"],
            "unit":      data["unit"],
            "flag":      data["flag"],
        })

    print(f"\n📊 Test contains {len(lab_results)} metrics:")
    for r in lab_results:
        print(f"   • {r['test_name']:15} = {r['value']} {r['unit']:8} [{r['flag']}]")

    print("\n🎯 Triggering analysis workflow...")
    print("=" * 70 + "\n")

    state: AgentState = {
        "request_type":   "blood_test_analysis",
        "patient_id":     patient_id,
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


__all__ = ["AgentState", "build_system", "simulate_new_lab_result", "analyze_existing_test"]