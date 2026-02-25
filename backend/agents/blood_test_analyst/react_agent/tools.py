from backend.tools.medline_test_rag import create_medline_test_rag
from langchain_core.tools import tool
from typing import Dict, Any, Literal
from backend.agents.blood_test_analyst.data_mockups.patients_mockup import *
from backend.agents.blood_test_analyst.data_mockups.reference_ranges_mockup import *
from backend.supabase.supabase_client import (
    get_patients_summary,
    get_patient_lab_history,
    fetch_patient_by_id,
)


@tool
def search_medical_knowledge(query: str) -> dict:
    """
    Search medical literature for causes, conditions, and treatments.

    Uses RAG system with MedlinePlus database.

    Args:
        query: Medical question (e.g., "causes of low hemoglobin")

    Returns:
        Medical information from trusted sources
    """
    rag = create_medline_test_rag()
    results = rag.answer_question(query)

    # BUG FIX: removed nested @tool definition that shadowed this function
    # and caused it to always return None. The return is now here, at the
    # correct indentation level inside the outer (real) tool.
    return {
        "rag_sys_prompt":  results.get("llm_system_prompt", ""),
        "rag_user_prompt": results.get("llm_user_prompt", ""),
        "answer": f"Medical knowledge about '{query}': {results['answer']}"
    }


@tool
def get_patient_history(patient_id: str) -> Dict[str, Any]:
    """
    Retrieve complete patient medical history.

    Gets demographics, medications, chronic conditions,
    lab history, and recent clinical notes.

    Args:
        patient_id: Patient identifier (e.g., "P001", "P002", "P003")

    Returns:
        Dictionary with patient data
    """
    return fetch_patient_by_id(patient_id)


@tool
def check_reference_range(
        test_name: str,
        value: float,
        patient_sex: Literal["M", "F"],
        patient_age: int
) -> Dict[str, Any]:
    """
    Check if a lab value is within normal reference range for patient demographics.
    Provides clinical interpretation including severity assessment.

    Args:
        test_name:   Name of lab test (e.g., "Hemoglobin", "Glucose", "Creatinine")
        value:       Test result value
        patient_sex: Patient biological sex
        patient_age: Patient age in years

    Returns:
        Dictionary with:
        - status: normal/borderline_low/borderline_high/abnormal_low/abnormal_high/critical
        - severity: normal/borderline/abnormal/critical
        - flag: normal/low/high/critical_low/critical_high
        - reference_range: "low-high unit"
        - interpretation: Human-readable interpretation
        - clinical_significance: What this means clinically
    """
    ref_range = get_reference_range(test_name, patient_sex, patient_age)

    if ref_range is None:
        return {
            "error": f"No reference range available for {test_name}",
            "available_tests": ["Hemoglobin", "Glucose", "Creatinine", "HbA1c"]
        }

    classification = classify_value(value, ref_range, test_name)

    result = {
        "test_name": test_name,
        "value": value,
        "unit": ref_range["unit"],
        "reference_range": f"{ref_range['low']}-{ref_range['high']} {ref_range['unit']}",
        "patient_demographics": f"{patient_age}yo {patient_sex}",
        **classification
    }
    return result


# ── Helpers ────────────────────────────────────────────────────────────────

def get_reference_range(test_name: str, sex: str, age: int = None):
    """
    Get reference range for a specific test.

    Args:
        test_name: Name of the lab test
        sex:       Patient sex ("M" or "F")
        age:       Patient age (currently using adult ranges, can be extended)

    Returns:
        Dictionary with reference range info or None if not found
    """
    if test_name not in REFERENCE_RANGES:
        return None
    if sex not in REFERENCE_RANGES[test_name]:
        return None
    age_category = "adult"
    if age_category not in REFERENCE_RANGES[test_name][sex]:
        return None
    return REFERENCE_RANGES[test_name][sex][age_category]


def classify_value(value: float, ref_range: dict, test_name: str = None) -> dict:
    """
    Classify a lab value based on reference range.

    Args:
        value:     The test result value
        ref_range: Reference range dictionary
        test_name: Optional test name for special handling

    Returns:
        Classification with status, severity, and interpretation
    """
    low           = ref_range["low"]
    high          = ref_range["high"]
    critical_low  = ref_range.get("critical_low")
    critical_high = ref_range.get("critical_high")

    if critical_low and value < critical_low:
        status, severity, flag = "critical_low", "critical", "critical_low"
    elif critical_high and value > critical_high:
        status, severity, flag = "critical_high", "critical", "critical_high"
    elif value < low:
        percent_below = ((low - value) / low) * 100
        if percent_below > 20:
            status, severity, flag = "abnormal_low", "abnormal", "low"
        else:
            status, severity, flag = "borderline_low", "borderline", "low"
    elif value > high:
        percent_above = ((value - high) / high) * 100
        if percent_above > 20:
            status, severity, flag = "abnormal_high", "abnormal", "high"
        else:
            status, severity, flag = "borderline_high", "borderline", "high"
    else:
        status, severity, flag = "normal", "normal", "normal"

    # Special handling for glucose (prediabetic / diabetic ranges)
    if test_name == "Glucose" and value >= ref_range.get("diabetic_threshold", 999):
        additional_info = "Diabetic range (≥126 mg/dL fasting)"
    elif test_name == "Glucose" and value >= ref_range.get("prediabetic_threshold", 999):
        additional_info = "Prediabetic range (100-125 mg/dL fasting)"
    else:
        additional_info = None

    return {
        "status":          status,
        "severity":        severity,
        "flag":            flag,
        "interpretation":  INTERPRETATION_MESSAGES[status],
        "additional_info": additional_info,
    }