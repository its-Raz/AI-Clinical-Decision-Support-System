"""
Nodes for the Skin Care Classifier Agent.

Graph flow:  [classify_node]  →  [report_node]  →  END

- classify_node : Invokes the YOLO tool and stores raw results in state.
- report_node   : Calls the LLM to generate the final patient-facing report.
"""

from langchain_core.messages import HumanMessage
from .tools import classify_skin_lesion
from .prompts import REPORT_PROMPT_TEMPLATE


def classify_node(state: dict, llm) -> dict:
    """
    Invokes the YOLO classification tool on the image_path from state.
    Updates: vision_results, messages.
    """
    image_path = state["image_path"]
    print(f"🔍 Classifying image: {image_path}")

    vision_results = classify_skin_lesion.invoke({"image_path": image_path})

    # e.g. {"bbox": [...], "raw_class": "High_Urgency_mel",
    #        "label": "High Urgency", "finding": "possible melanoma", "conf": 0.91}
    print(f"   ✅ Result: label={vision_results.get('label')} | "
          f"finding={vision_results.get('finding')} | conf={vision_results.get('conf')}")

    tool_msg = {
        "role": "tool",
        "name": "classify_skin_lesion",
        "content": str(vision_results),
    }
    return {
        "vision_results": vision_results,
        "messages": [tool_msg],
    }


def report_node(state: dict, llm) -> dict:
    """
    Generates a professional clinical report from vision_results using the LLM.
    Updates: final_report, messages.
    """
    vision_results = state.get("vision_results", {})
    patient_id     = state.get("patient_id", "Unknown")

    # Handle tool error gracefully
    if "error" in vision_results:
        error_report = (
            f"⚠️ The skin lesion classification could not be completed. "
            f"Reason: {vision_results['error']}. "
            "Please ensure the image is valid and retry, or consult a physician directly."
        )
        return {
            "final_report": error_report,
            "messages": [{"role": "assistant", "content": error_report}],
        }

    label     = vision_results["label"]        # "High Urgency" or "Low Urgency"
    finding   = vision_results["finding"]      # "possible melanoma" or "nevus (benign mole)"
    conf      = vision_results["conf"]
    bbox      = vision_results["bbox"]
    conf_pct  = round(conf * 100, 1)
    is_high   = label == "High Urgency"        # True for High_Urgency_mel

    urgency_addendum = (
        " IMMEDIATELY — please seek medical attention today or within 24 hours"
        if is_high
        else " at your earliest convenience for a routine evaluation"
    )

    prompt = REPORT_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        label=label,
        finding=finding,
        conf_pct=conf_pct,
        bbox=[round(v, 1) for v in bbox],
        urgency_addendum=urgency_addendum,
    )

    print(f"📝 Generating report | Patient: {patient_id} | {label} ({finding}) @ {conf_pct}%")
    response = llm.invoke([HumanMessage(content=prompt)])
    report   = response.content

    return {
        "final_report": report,
        "messages": [{"role": "assistant", "content": report}],
    }