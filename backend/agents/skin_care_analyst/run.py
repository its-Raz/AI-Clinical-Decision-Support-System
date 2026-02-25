"""
backend/agents/skin_care_analyst/run.py

LangGraph node wrapper for the Skin Care Analyst.
Called by the Manager graph when request_type == "image_lesion_analysis".
"""

import os
import logging

log = logging.getLogger(__name__)

# ── Demo image — used whenever no real image_path is provided ─────────────
# This allows the /api/execute endpoint to demonstrate skin-care analysis
# without requiring an actual file upload.
_DEMO_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "demo_data", "ISIC_0024500.jpg"
)


def run_skin_care_analyst(state: dict) -> dict:
    """
    LangGraph node — runs the SkinCareAgent and populates vision_insights.

    Reads:   state["image_path"]  (falls back to demo image if absent/None)
    Writes:  state["vision_results"]  (raw YOLO output)
             state["vision_insights"] (clinical summary for manager to reshape)
             state["messages"]        (trace entries)
             state["steps"]           (structured step objects for API trace)

    Step trace produced (via operator.add inside SkinCareAgent sub-graph):
        1. SkinCareAnalyst/Tool:classify_skin_lesion  — YOLO tool call + result
        2. SkinCareAnalyst/ReportNode                 — LLM prompt + clinical report
    """
    log.debug("run_skin_care_analyst() called")
    print("\n" + "─" * 50)
    print("🩺 [run_skin_care_analyst] ENTER")

    from backend.agents.skin_care_analyst.agent import SkinCareAgent

    patient_id = state.get("patient_id", "unknown")
    image_path = state.get("image_path")

    # ── Demo fallback ─────────────────────────────────────────────────────
    # If no image was supplied (e.g. plain-text API call), use the bundled
    # demo image so the full pipeline can still be demonstrated end-to-end.
    using_demo = False
    if not image_path:
        image_path = _DEMO_IMAGE_PATH
        using_demo = True
        log.info("run_skin_care_analyst: no image_path provided — using demo image")
        print(f"   ℹ️  No image_path in state — falling back to demo image")

    print(f"   patient_id  : {patient_id}")
    print(f"   image_path  : {image_path}")
    print(f"   using_demo  : {using_demo}")

    # ── Sanity check: demo image must exist on disk ───────────────────────
    if not os.path.exists(image_path):
        msg = (
            f"Image not found: {image_path}. "
            "Please ensure demo_data/ISIC_0024500.jpg is present."
        )
        log.error("run_skin_care_analyst: %s", msg)
        print(f"   ❌ ERROR: {msg}")
        return {
            "vision_insights": f"Error: {msg}",
            "messages": [{
                "role":    "system",
                "content": f"[Skin Care Analyst] ❌ {msg}",
            }],
            "steps": [{
                "module":   "Skin Care Analyst Agent",
                "prompt":   f"patient_id={patient_id}, image_path={image_path}",
                "response": f"ERROR: {msg}",
            }],
        }

    trace_msgs = [{
        "role":    "system",
        "content": (
            f"[Skin Care Analyst] Analyzing skin lesion for {patient_id} "
            f"({'demo image' if using_demo else image_path}) …"
        ),
    }]

    try:
        # ── Run the SkinCareAgent ─────────────────────────────────────────
        agent = SkinCareAgent()

        # Build mini-state with the resolved image_path.
        # "steps": [] must be initialised so operator.add can accumulate
        # the steps written by classify_node and report_node inside the
        # SkinCareAgent sub-graph.
        mini_state = {
            "request_type":              "image_lesion_analysis",
            "patient_id":                patient_id,
            "image_path":                image_path,
            "lab_result":                None,
            "lab_insights":              None,
            "vision_results":            None,
            "vision_insights":           None,
            "messages":                  [],
            "next_step":                 "",
            "final_report":              None,
            "steps":                     [],   # operator.add accumulates here
            "raw_user_input":            None,
            "router_proposed_category":  None,
            "router_score":              None,
            "router_confidence":         None,
            "evidence_insights":         None,
        }

        result_state = agent.run(mini_state)

        vision_results  = result_state.get("vision_results", {})
        vision_insights = result_state.get("final_report", "")

        # ── Collect steps written by the sub-graph nodes ──────────────────
        # classify_node  → SkinCareAnalyst/Tool:classify_skin_lesion
        # report_node    → SkinCareAnalyst/ReportNode
        # operator.add has already accumulated them in result_state["steps"]
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
            "steps":           agent_steps,   # operator.add appends to graph-level steps
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