REACT_SYSTEM_PROMPT = """You are a medical research assistant gathering information to analyze laboratory test results. You have access to tools to retrieve patient history, check reference ranges, and search medical knowledge."""

REACT_PROMPT_TEMPLATE = """Analyze laboratory test results by gathering relevant clinical information.

Available tools:
- get_patient_history[patient_id]: Retrieve patient demographics, chronic conditions, lab history, and clinical notes
- check_reference_range[test_name, value, patient_sex, patient_age]: Determine if lab value is normal/abnormal for patient demographics
- search_medical_knowledge[query]: Search medical literature for causes, conditions, and clinical significance

Use this exact format:

Lab Result: [test name and value]
Thought 1: [what information do I need?]
Action 1: [tool_name[parameters]]
Observation 1: [tool result]
Thought 2: [analyze the result, decide next step]
Action 2: [tool_name[parameters]] OR Finish[summary of all findings]
...

EXAMPLES:

Lab Result: Patient P001 - Hemoglobin 9.8 g/dL (Flag: low)
Thought 1: I need to get patient history first to understand their baseline and any chronic conditions that might explain low hemoglobin.
Action 1: get_patient_history[P001]
Observation 1: Patient is Sarah Mitchell, 45yo female. Chronic conditions: Iron deficiency anemia. Recent notes: "Transitioned to vegetarian diet 11 months ago", "Heavy menstrual periods", "Started iron supplementation 10 weeks ago with minimal improvement". Lab history shows declining hemoglobin: 13.2 → 13.0 → 12.5 → 11.8 → 11.0 → 10.2 → 9.8 g/dL over 18 months.
Thought 2: The patient has a clear declining trend in hemoglobin correlating with dietary change. I should check if 9.8 g/dL is significantly abnormal for a 45-year-old female.
Action 2: check_reference_range[Hemoglobin, 9.8, F, 45]
Observation 2: Status: abnormal_low. Reference range: 12.0-15.5 g/dL. Severity: abnormal. Value is 18% below normal range for 45yo female. Interpretation: Anemia present.
Thought 3: Now I need to understand the medical causes of declining hemoglobin in the context of vegetarian diet and heavy periods.
Action 3: search_medical_knowledge[causes of iron deficiency anemia in vegetarian women]
Observation 3: Iron deficiency anemia develops when dietary iron intake is insufficient (vegetarian/vegan diets lack heme iron), iron absorption is impaired, or blood loss occurs (heavy menstruation). Diagnosis requires low hemoglobin + low ferritin. Treatment: iron supplementation (oral or IV if not responding) + address underlying cause (dietary counseling, menstrual management).
Thought 4: I now have sufficient information: declining hemoglobin trend (13.2→9.8), clear causes (vegetarian diet + heavy periods), patient already on iron supplements with minimal response, and medical context about iron deficiency. This is enough to analyze.
Action 4: Finish[Patient has progressive iron deficiency anemia (Hgb 9.8, 18% below normal) over 18 months correlating with vegetarian diet transition and heavy menstrual periods. Already on oral iron supplementation for 10 weeks without adequate response. Medical literature confirms dietary insufficiency and blood loss as primary causes. May need IV iron therapy.]

Lab Result: Patient P003 - Glucose 178 mg/dL (Flag: high)
Thought 1: This is a significantly elevated glucose. I need patient history to see if this is chronic hyperglycemia or an acute event.
Action 1: get_patient_history[P003]
Observation 1: Patient is Emily Rodriguez, 35yo female. No chronic conditions. Recent notes: "Patient presented with influenza (fever 102.8°F, positive flu test)", "Labs drawn while acutely ill", "Advised to repeat labs after recovery". Lab history shows stable normal values for 6 tests, then glucose spike to 178 mg/dL during illness, then returned to 87 mg/dL after recovery.
Thought 2: This appears to be stress hyperglycemia during acute illness, not diabetes. The glucose normalized after recovery. I should verify the reference range to confirm 178 is abnormal.
Action 2: check_reference_range[Glucose, 178, F, 35]
Observation 2: Status: abnormal_high. Reference range: 70-100 mg/dL. Severity: abnormal. Value is 78% above normal range.
Thought 3: The glucose was very elevated (178 mg/dL, 78% above normal) but only during acute influenza infection. I should search for information about stress hyperglycemia during illness.
Action 3: search_medical_knowledge[stress hyperglycemia during acute infection]
Observation 3: Stress hyperglycemia can occur during acute illness, infection, or physiological stress without underlying diabetes. Counter-regulatory hormones (cortisol, epinephrine) increase glucose production. Typically resolves after illness recovery. Patients should have follow-up glucose testing to confirm resolution and rule out unmasked diabetes.
Thought 4: Complete picture: acute stress hyperglycemia during influenza (178 mg/dL), normalized after recovery (87 mg/dL), no diabetes history, medical literature confirms this is a known phenomenon. Sufficient information gathered.
Action 4: Finish[Patient had transient stress hyperglycemia (glucose 178 mg/dL, 78% above normal) during acute influenza infection. All previous values normal (85-89 mg/dL). Glucose normalized to 87 mg/dL after recovery, confirming stress response rather than diabetes. Medical literature supports this as physiological response to acute illness. No evidence of underlying diabetes mellitus.]

Now analyze this lab result:

{input}"""