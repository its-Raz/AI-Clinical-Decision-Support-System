from typing import TypedDict, Annotated, List, Optional, Union
import operator


class AgentState(TypedDict):
    # --- Request Type Flag ---
    # Determines the workflow path: "blood_test_analysis" or "image_lesion_analysis"
    request_type: str

    # --- Patient Identification ---
    patient_id: str

    # --- Blood Test Specific Fields ---
    # Populated only if request_type is "blood_test_analysis"
    lab_result: Optional[List[dict]]  # Raw laboratory data from the test_images
    lab_insights: Optional[str]  # Insights generated via RAG and the Lab Analyst

    # --- Image Analysis Specific Fields ---
    # Populated only if request_type is "image_lesion_analysis"
    image_path: Optional[str]  # Local or cloud path to the uploaded image
    vision_results: Optional[dict]  # YOLO output: Bounding Box, Label, and Confidence score

    # --- Orchestration & Flow Control ---
    # Accumulative chat history. Annotated with operator.add to append instead of overwrite
    messages: Annotated[List[dict], operator.add]

    # Determines which node (agent) the router should invoke next
    next_step: str

    # The final synthesized report formatted for the physician or patient
    final_report: Optional[str]