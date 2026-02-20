"""
backend/data — Patient data access layer.

Supabase-based patient database interface.
"""

from .supabase_client import (
    get_supabase_client,
    fetch_all_patients,
    fetch_patient_by_id,
    add_lab_result,
    get_patients_with_low_hemoglobin,
    get_first_patient,
    get_patients_summary,
    get_patient_lab_history,
)

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