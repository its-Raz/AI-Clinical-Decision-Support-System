# Purpose: Define the global state model that flows through ALL agents
# Contains:
#
# One Pydantic model: ClinicalState
# All fields needed by Manager, Analyst, and future agents
# Input fields (patient_id, lab_result)
# Output fields (diagnosis, patient_responses, appointment_info)
# Metadata fields (timestamps, iteration counts)
#
# Why separate: Both Manager and Analyst need the same state definition