"""
backend/data/supabase_client.py

Supabase client for accessing patient data from the database.
Replaces the mockup patient data with real database queries.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Lazy import - only load if credentials are present
_supabase_client = None


def get_supabase_client():
    """Get or create Supabase client instance."""
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY "
            "in your .env file."
        )

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        print(f"✅ Supabase client initialized: {url[:30]}...")
        return _supabase_client
    except ImportError:
        raise ImportError(
            "supabase package not found. Install it with: pip install supabase --break-system-packages"
        )


def fetch_all_patients() -> List[Dict]:
    """
    Fetch all patients from Supabase.

    Returns:
        List of patient dicts with structure:
        {
            "id": "P001",
            "name": "Olivia Davis",
            "age": 45,
            "sex": "F",
            "chronic_conditions": ["Hypertension", "Type 2 Diabetes"],
            "lab_history": [{date, Hemoglobin, Glucose, Creatinine}, ...],
            "recent_notes": ["2024-01-15: Patient stable..."]
        }
    """
    client = get_supabase_client()

    try:
        response = client.table("patients").select("*").execute()

        if not response.data:
            print("⚠️  No patients found in Supabase")
            return []

        print(f"📊 Loaded {len(response.data)} patients from Supabase")
        return response.data

    except Exception as e:
        print(f"❌ Error fetching patients from Supabase: {e}")
        raise


def fetch_patient_by_id(patient_id: str) -> Optional[Dict]:
    """
    Fetch a single patient by ID.

    Args:
        patient_id: Patient ID like "P001"

    Returns:
        Patient dict or None if not found
    """
    client = get_supabase_client()

    try:
        response = client.table("patients").select("*").eq("id", patient_id).execute()

        if not response.data:
            print(f"⚠️  Patient {patient_id} not found in Supabase")
            return None

        return response.data[0]

    except Exception as e:
        print(f"❌ Error fetching patient {patient_id}: {e}")
        raise


def add_lab_result(patient_id: str, lab_entry: Dict) -> bool:
    """
    Add a new lab result to a patient's lab_history.

    Args:
        patient_id: Patient ID like "P001"
        lab_entry: Lab entry dict with structure:
            {
                "date": "2026-02-18",
                "Hemoglobin": {"value": 14.1, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 95, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 1.0, "unit": "mg/dL", "flag": "normal"}
            }

    Returns:
        True if successful
    """
    client = get_supabase_client()

    try:
        # Fetch current patient
        patient = fetch_patient_by_id(patient_id)
        if not patient:
            return False

        # Append new lab entry
        current_history = patient.get("lab_history", [])
        current_history.append(lab_entry)

        # Update in Supabase
        response = client.table("patients").update(
            {"lab_history": current_history}
        ).eq("id", patient_id).execute()

        print(f"✅ Added lab result for {patient_id}")
        return True

    except Exception as e:
        print(f"❌ Error adding lab result: {e}")
        return False


def get_patients_with_low_hemoglobin(threshold: float = 10.5) -> List[Dict]:
    """
    Find patients with hemoglobin below threshold in their most recent lab.

    Args:
        threshold: Hemoglobin threshold in g/dL

    Returns:
        List of dicts: [{"id": "P001", "name": "...", "hgb": 9.8, "date": "..."}, ...]
    """
    all_patients = fetch_all_patients()
    at_risk = []

    for patient in all_patients:
        lab_history = patient.get("lab_history", [])
        if not lab_history:
            continue

        # Get most recent lab
        latest = lab_history[-1]
        hgb_data = latest.get("Hemoglobin", {})
        hgb_value = hgb_data.get("value")

        if hgb_value and hgb_value < threshold:
            at_risk.append({
                "id": patient["id"],
                "name": patient["name"],
                "hgb": hgb_value,
                "date": latest.get("date", "Unknown"),
            })

    return at_risk


def get_first_patient() -> Optional[Dict]:
    """
    Get the first patient from the database (for demo purposes).

    Returns:
        Patient dict or None if no patients exist
    """
    client = get_supabase_client()

    try:
        response = client.table("patients").select("*").order("id").limit(1).execute()

        if not response.data:
            print("⚠️  No patients found in Supabase")
            return None

        patient = response.data[0]
        print(f"📋 Using demo patient: {patient['id']} ({patient['name']})")
        return patient

    except Exception as e:
        print(f"❌ Error fetching first patient: {e}")
        raise


def get_patients_summary() -> List[Dict]:
    """
    Get all patients with summary information for UI selection.

    Returns:
        List of dicts: [{"id": "P001", "name": "...", "age": 45, "test_count": 7}, ...]
    """
    all_patients = fetch_all_patients()

    summary = []
    for patient in all_patients:
        lab_history = patient.get("lab_history", [])
        summary.append({
            "id": patient["id"],
            "name": patient["name"],
            "age": patient.get("age", "N/A"),
            "sex": patient.get("sex", "N/A"),
            "test_count": len(lab_history),
        })

    # Sort by ID for consistent display
    summary.sort(key=lambda x: x["id"])
    return summary


def get_patient_lab_history(patient_id: str) -> List[Dict]:
    """
    Get all lab results for a specific patient.

    Args:
        patient_id: Patient ID like "P001"

    Returns:
        List of lab result dicts with date, metrics, flags
    """
    patient = fetch_patient_by_id(patient_id)
    if not patient:
        return []

    return patient.get("lab_history", [])


__all__ = [
    "get_supabase_client",
    "fetch_all_patients",
    "fetch_patient_by_id",
    "add_lab_result",
    "get_patients_with_low_hemoglobin",
    "get_first_patient",
    "get_patients_summary",
    "get_patient_lab_history",
]