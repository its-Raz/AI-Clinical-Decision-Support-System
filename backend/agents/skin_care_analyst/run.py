"""
backend/agents/skin_care_analyst/run.py

LangGraph node wrapper for the Skin Care Analyst.
Called by the Manager graph when request_type == "image_lesion_analysis".
"""

import logging

log = logging.getLogger(__name__)


def run_skin_care_analyst(state: dict) -> dict:
    """
    LangGraph node — runs the SkinCareAgent and populates vision_insights.

    Reads:   state["image_path"]
    Writes:  state["vision_results"]  (raw YOLO output)
             state["vision_insights"] (clinical summary for manager to reshape)
             state["messages"]        (trace entries)
             state["steps"]           (structured step objects for API)
    """
    log.debug("run_skin_care_analyst() called")
    print("\n" + "─" * 50)
    print("🩺 [run_skin_care_analyst] ENTER")

    from backend.agents.skin_care_analyst.agent import SkinCareAgent

    patient_id = state.get("patient_id", "unknown")
    image_path = state.get("image_path")

    print(f"   patient_id : {patient_id}")
    print(f"   image_path : {image_path}")

    if not image_path:
        log.error("run_skin_care_analyst: image_path is None")
        print("   ❌ ERROR: image_path not provided")
        return {
            "vision_insights": "Error: No image provided for analysis.",
            "messages": [{
                "role":    "system",
                "content": "[Skin Care Analyst] ❌ Error: image_path missing",
            }],
            "steps": [{
                "module":   "SkinCareAnalyst",
                "prompt":   f"patient_id={patient_id}, image_path=None",
                "response": "ERROR: image_path not provided",
            }],
        }

    trace_msgs = [{
        "role":    "system",
        "content": f"[Skin Care Analyst] Analyzing skin lesion for {patient_id} …",
    }]

    try:
        # ── Run the SkinCareAgent ─────────────────────────────────────
        agent = SkinCareAgent()

        # Build mini-state — includes "steps": [] so operator.add
        # can accumulate steps written by classify_node and report_node
        mini_state = {
            "request_type":    "image_lesion_analysis",
            "patient_id":      patient_id,
            "image_path":      image_path,
            "lab_result":      None,
            "lab_insights":    None,
            "vision_results":  None,
            "vision_insights": None,
            "messages":        [],
            "next_step":       "",
            "final_report":    None,
            "steps":           [],   # ← must be initialised so operator.add works
            # router fields (not used by skin care agent internally)
            "raw_user_input":             None,
            "router_proposed_category":   None,
            "router_score":               None,
            "router_confidence":          None,
            "evidence_insights":          None,
        }

        result_state = agent.run(mini_state)

        vision_results  = result_state.get("vision_results", {})
        vision_insights = result_state.get("final_report", "")

        # ── Extract steps accumulated inside the SkinCareAgent graph ──
        # classify_node wrote SkinCareAnalyst/Tool:classify_skin_lesion
        # report_node  wrote SkinCareAnalyst/ReportNode
        # operator.add accumulated them in result_state["steps"]
        agent_steps = result_state.get("steps", [])

        log.info(
            "run_skin_care_analyst: label=%s, conf=%.2f, insights=%d chars, steps=%d",
            vision_results.get("label", "N/A"),
            vision_results.get("conf", 0),
            len(vision_insights),
            len(agent_steps),
        )

        print(f"   ✅ YOLO label    : {vision_results.get('label', 'N/A')}")
        print(f"   ✅ Confidence    : {vision_results.get('conf', 0):.2%}")
        print(f"   ✅ Insights      : {len(vision_insights)} chars")
        print(f"   ✅ Steps captured: {len(agent_steps)}")

        trace_msgs.append({
            "role":    "system",
            "content": (
                f"[Skin Care Analyst] Analysis complete. "
                f"Label: {vision_results.get('label', 'N/A')}, "
                f"Confidence: {vision_results.get('conf', 0):.2%}."
            ),
        })

        print("─" * 50)
        return {
            "vision_results":  vision_results,
            "vision_insights": vision_insights,
            "messages":        trace_msgs,
            "steps":           agent_steps,   # ← operator.add appends to graph-level steps
        }

    except Exception as e:
        log.error("run_skin_care_analyst: FAILED — %s", e, exc_info=True)
        print(f"   ❌ ERROR: {e}")
        print("─" * 50)

        trace_msgs.append({
            "role":    "system",
            "content": f"[Skin Care Analyst] ❌ Error: {e}",
        })

        return {
            "vision_insights": f"Error during skin lesion analysis: {str(e)}",
            "messages":        trace_msgs,
            "steps": [{
                "module":   "SkinCareAnalyst",
                "prompt":   f"patient_id={patient_id}, image_path={image_path}",
                "response": f"ERROR: {e}",
            }],
        }