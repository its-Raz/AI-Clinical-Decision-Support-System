# Patient P001 - Sarah Mitchell (The Anemia Case)
#
# Hemoglobin declines from 13.2 → 9.8 g/dL over 18 months
# Caused by vegetarian diet transition + heavy menstrual periods
# Recently started iron supplementation
#
# Patient P002 - Robert Chen (The Prediabetic)
#
# Glucose rises from 92 → 118 mg/dL (prediabetic range)
# Sedentary lifestyle, weight gain, family history
# Creatinine trending up (1.08 → 1.23) showing early kidney stress
# Recent lifestyle changes showing slight improvement
#
# Patient P003 - Emily Rodriguez (The Anomaly)
#
# All values stable and normal for 6 tests
# Glucose spikes to 178 mg/dL during acute influenza infection
# Returns to baseline (87 mg/dL) after recovery
# Perfect test case for anomaly detection algorithms

MOCK_PATIENTS = {
    "P001": {
        "name": "Sarah Mitchell",
        "age": 45,
        "sex": "F",
        "chronic_conditions": ["Iron deficiency anemia"],
        "lab_history": [
            {
                "date": "2024-08-15",
                "Hemoglobin": {"value": 13.2, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 88, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.85, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2024-11-15",
                "Hemoglobin": {"value": 13.0, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 86, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.87, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-02-15",
                "Hemoglobin": {"value": 12.5, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 90, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.84, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-05-15",
                "Hemoglobin": {"value": 11.8, "unit": "g/dL", "flag": "low"},
                "Glucose": {"value": 89, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.86, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-08-15",
                "Hemoglobin": {"value": 11.0, "unit": "g/dL", "flag": "low"},
                "Glucose": {"value": 92, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.83, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-11-15",
                "Hemoglobin": {"value": 10.2, "unit": "g/dL", "flag": "low"},
                "Glucose": {"value": 87, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.85, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2026-02-15",
                "Hemoglobin": {"value": 9.8, "unit": "g/dL", "flag": "low"},
                "Glucose": {"value": 91, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.84, "unit": "mg/dL", "flag": "normal"}
            }
        ],
        "recent_notes": [
            "2025-01-10: Patient reports transitioning to vegetarian diet approximately 11 months ago for ethical reasons.",
            "2025-06-20: Complaints of persistent fatigue, difficulty concentrating, and occasional dizziness. Hemoglobin trending downward.",
            "2025-09-05: Patient reports heavy menstrual periods. Referred to gynecology for evaluation.",
            "2025-11-28: Severe fatigue interfering with daily activities. Started on iron supplementation (ferrous sulfate 325mg daily) and B12.",
            "2026-02-15: Patient compliant with iron supplements for 10 weeks. Reports slight improvement in energy but still fatigued. Hemoglobin has not yet responded significantly. Dietary counseling provided for iron-rich plant foods. Consider IV iron if no improvement in 6 weeks."
        ]
    },

    "P002": {
        "name": "Robert Chen",
        "age": 58,
        "sex": "M",
        "chronic_conditions": ["Hypertension", "Hyperlipidemia", "Prediabetes"],
        "lab_history": [
            {
                "date": "2024-08-15",
                "Hemoglobin": {"value": 14.8, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 92, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 1.08, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2024-11-15",
                "Hemoglobin": {"value": 15.1, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 96, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 1.12, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-02-15",
                "Hemoglobin": {"value": 14.9, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 101, "unit": "mg/dL", "flag": "high"},
                "Creatinine": {"value": 1.15, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-05-15",
                "Hemoglobin": {"value": 15.0, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 106, "unit": "mg/dL", "flag": "high"},
                "Creatinine": {"value": 1.18, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-08-15",
                "Hemoglobin": {"value": 14.7, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 110, "unit": "mg/dL", "flag": "high"},
                "Creatinine": {"value": 1.21, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-11-15",
                "Hemoglobin": {"value": 15.2, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 118, "unit": "mg/dL", "flag": "high"},
                "Creatinine": {"value": 1.25, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2026-02-15",
                "Hemoglobin": {"value": 14.8, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 115, "unit": "mg/dL", "flag": "high"},
                "Creatinine": {"value": 1.23, "unit": "mg/dL", "flag": "normal"}
            }
        ],
        "recent_notes": [
            "2024-09-12: Patient reports sedentary lifestyle, works from home as software engineer. Weight: 218 lbs (up 12 lbs from last year).",
            "2025-02-20: Patient reports increased thirst and occasional blurred vision. Fasting glucose now 101 mg/dL. Discussed prediabetes risk.",
            "2025-05-18: Weight now 224 lbs. HbA1c ordered: 5.8% (prediabetic range). Lifestyle modification counseling provided.",
            "2025-08-20: Patient admits poor compliance with diet and exercise recommendations. Reports frequent urination at night. Glucose continues to rise.",
            "2025-11-20: Glucose 118 mg/dL, approaching diabetic threshold. Patient motivated after uncle diagnosed with Type 2 diabetes. Enrolled in diabetes prevention program.",
            "2026-02-15: Slight improvement in glucose (115 mg/dL, down from 118). Patient reports walking 20 minutes daily for past 6 weeks and reduced carbohydrate intake. Weight down 4 lbs to 220 lbs. Creatinine trending upward - monitoring kidney function. Encouraged to continue lifestyle changes. Recheck in 3 months with HbA1c."
        ]
    },

    "P003": {
        "name": "Emily Rodriguez",
        "age": 35,
        "sex": "F",

        "chronic_conditions": [],
        "lab_history": [
            {
                "date": "2024-08-15",
                "Hemoglobin": {"value": 13.6, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 87, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.82, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2024-11-15",
                "Hemoglobin": {"value": 13.7, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 85, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.84, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-02-15",
                "Hemoglobin": {"value": 13.5, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 89, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.83, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-05-15",
                "Hemoglobin": {"value": 13.8, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 86, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.85, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-08-15",
                "Hemoglobin": {"value": 13.4, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 88, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.82, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2025-11-15",
                "Hemoglobin": {"value": 12.8, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 178, "unit": "mg/dL", "flag": "high"},
                "Creatinine": {"value": 0.96, "unit": "mg/dL", "flag": "normal"}
            },
            {
                "date": "2026-02-15",
                "Hemoglobin": {"value": 13.7, "unit": "g/dL", "flag": "normal"},
                "Glucose": {"value": 87, "unit": "mg/dL", "flag": "normal"},
                "Creatinine": {"value": 0.84, "unit": "mg/dL", "flag": "normal"}
            }
        ],
        "recent_notes": [
            "2024-08-15: Annual wellness visit. Patient healthy, exercises regularly (runs 3x/week), balanced diet. No complaints.",
            "2025-02-15: Routine follow-up. All labs normal. Patient training for half-marathon.",
            "2025-05-15: Routine labs. Patient in excellent health.",
            "2025-08-15: Annual physical. No concerns. Continue current lifestyle.",
            "2025-11-08: Patient presented to urgent care with influenza-like illness: high fever (102.8°F), body aches, cough, fatigue. Rapid flu test positive for Influenza A. Prescribed oseltamivir.",
            "2025-11-15: Labs drawn while acutely ill with influenza. Glucose significantly elevated at 178 mg/dL (stress hyperglycemia). Hemoglobin slightly decreased likely due to acute illness and dehydration. Creatinine mildly elevated, consistent with dehydration. Patient advised to increase fluid intake and rest. Repeat labs in 3 months once recovered.",
            "2026-02-15: Patient fully recovered from influenza. All lab values returned to normal baseline. Glucose 87 mg/dL, hemoglobin 13.7 g/dL, creatinine 0.84 mg/dL. November spike confirmed as stress response to acute infection. No evidence of diabetes. Patient feeling well, resumed running schedule."
        ]
    }
}


# Utility functions for working with the mock data
def get_patient(patient_id):
    """Retrieve a specific patient's data."""
    return MOCK_PATIENTS.get(patient_id)