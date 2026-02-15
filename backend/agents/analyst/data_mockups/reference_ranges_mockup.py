"""
Clinical Laboratory Reference Ranges

Organized by test name with age/sex-specific ranges.
Based on standard clinical laboratory values.
"""

# Reference ranges by test name
# Structure: {test_name: {sex: {age_range: {low, high, unit, critical_low, critical_high}}}}

REFERENCE_RANGES = {
    "Hemoglobin": {
        "M": {
            "adult": {
                "low": 13.5,
                "high": 17.5,
                "unit": "g/dL",
                "critical_low": 7.0,  # Requires urgent intervention
                "critical_high": 20.0
            }
        },
        "F": {
            "adult": {
                "low": 12.0,
                "high": 15.5,
                "unit": "g/dL",
                "critical_low": 7.0,
                "critical_high": 18.0
            }
        }
    },

    "Glucose": {
        "M": {
            "adult": {
                "low": 70,
                "high": 100,  # Fasting normal
                "unit": "mg/dL",
                "critical_low": 40,  # Severe hypoglycemia
                "critical_high": 400,  # Severe hyperglycemia
                "prediabetic_threshold": 125,
                "diabetic_threshold": 126
            }
        },
        "F": {
            "adult": {
                "low": 70,
                "high": 100,
                "unit": "mg/dL",
                "critical_low": 40,
                "critical_high": 400,
                "prediabetic_threshold": 125,
                "diabetic_threshold": 126
            }
        }
    },

    "Creatinine": {
        "M": {
            "adult": {
                "low": 0.7,
                "high": 1.3,
                "unit": "mg/dL",
                "critical_low": 0.3,
                "critical_high": 5.0  # Severe kidney dysfunction
            }
        },
        "F": {
            "adult": {
                "low": 0.6,
                "high": 1.1,
                "unit": "mg/dL",
                "critical_low": 0.3,
                "critical_high": 5.0
            }
        }
    },

    # Add more tests as needed
    "HbA1c": {
        "M": {
            "adult": {
                "low": 4.0,
                "high": 5.6,  # Normal
                "unit": "%",
                "prediabetic_low": 5.7,
                "prediabetic_high": 6.4,
                "diabetic_threshold": 6.5,
                "critical_high": 14.0
            }
        },
        "F": {
            "adult": {
                "low": 4.0,
                "high": 5.6,
                "unit": "%",
                "prediabetic_low": 5.7,
                "prediabetic_high": 6.4,
                "diabetic_threshold": 6.5,
                "critical_high": 14.0
            }
        }
    }
}

# Clinical interpretation messages
INTERPRETATION_MESSAGES = {
    "normal": "Value is within normal reference range for patient demographics.",
    "borderline_low": "Value is slightly below normal range. May warrant monitoring.",
    "borderline_high": "Value is slightly above normal range. May warrant monitoring.",
    "abnormal_low": "Value is below normal range. Clinical evaluation recommended.",
    "abnormal_high": "Value is above normal range. Clinical evaluation recommended.",
    "critical_low": "⚠️ CRITICAL: Value is dangerously low. Immediate medical attention required.",
    "critical_high": "⚠️ CRITICAL: Value is dangerously high. Immediate medical attention required."
}


