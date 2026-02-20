# main.py
from backend.agents.global_state import AgentState
from run import run_react_agent
if __name__ == "__main__":
    # Instantiate the agent

    mock_global_state: AgentState = {
        "request_type": "blood_test_analysis",
        "patient_id": "PT-88392",
        "lab_result": [
            {
                "test_name": "Hemoglobin",
                "value": 9.5,
                "unit": "g/dL",
                "flag": "Low"
            },
            {
                "test_name": "Ferritin",
                "value": 12.0,
                "unit": "ng/mL",
                "flag": "Low"
            }
        ],
        "lab_insights": None,  # This is what your ReAct agent will eventually fill in
        "image_path": None,
        "vision_results": None,
        "vision_insights": None,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful medical routing assistant."
            },
            {
                "role": "user",
                "content": "I just got my lab results back and my Hemoglobin is 9.5 and Ferritin is 12.0. Both are marked as low. What are the common causes of this, and what dietary changes can help improve these levels?"
            }
        ],
        "next_step": "analyze_blood_test_evidence",
        "final_report": None
    }
    # Define a complex medical question


    # Run it!
    result = run_react_agent(mock_global_state)

    # Print the final LLM response
    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(result["messages"][-1].content)