"""
Manager node logic.

Two nodes:
  manager_node  – inspects state, logs routing decision, appends a trace message
  deliver_node  – takes lab_insights from the analyst and generates the
                  patient-facing final_report via LLM, using the system prompt
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from .prompts import MANAGER_SYSTEM_PROMPT, DELIVERY_PROMPT_TEMPLATE

log = logging.getLogger(__name__)


def manager_node(state: dict, llm) -> dict:
    """
    Inspect the incoming state and decide where to route.
    Sets `next_step` and appends a system trace message.
    Routing is deterministic — no LLM call needed here.
    The system prompt is applied in deliver_node where LLM is used.
    """
    log.debug("manager_node() called")
    print("\n" + "─" * 50)
    print("🧭 [manager_node] ENTER")

    request_type = state.get("request_type", "")
    patient_id   = state.get("patient_id", "unknown")
    lab_result   = state.get("lab_result") or []

    print(f"   patient_id   : {patient_id}")
    print(f"   request_type : {request_type}")
    print(f"   lab_result   : {len(lab_result)} metrics in batch")

    route_map = {
        "blood_test_analysis":   "blood_test_analyst",
        "image_lesion_analysis": "skin_care_analyst",
    }
    next_step = route_map.get(request_type, "unknown")

    if next_step == "unknown":
        print(f"   ⚠️  Unrecognised request_type '{request_type}' — fallback to deliver")
        log.warning("Unrecognised request_type: %s", request_type)

    print(f"   ✅ next_step  : {next_step}")
    print("─" * 50)

    trace_msg = {
        "role":    "system",
        "content": (
            f"[Manager] Patient {patient_id} | request_type='{request_type}' | "
            f"Routing → {next_step}. "
            f"Batch size: {len(lab_result)} metrics."
        ),
    }

    return {
        "next_step": next_step,
        "messages":  [trace_msg],
    }


def deliver_node(state: dict, llm) -> dict:
    """
    Receive the analyst's insights (lab_insights OR vision_insights),
    generate a patient-friendly final_report, and append it to messages.

    This node reshapes clinical output into empathetic, accessible language
    regardless of whether the source is blood test or skin care analysis.
    """
    log.debug("deliver_node() called")
    print("\n" + "─" * 50)
    print("💬 [deliver_node] ENTER")

    from .prompts import DELIVERY_PROMPT_TEMPLATE, DELIVERY_PROMPT_SKIN_CARE

    patient_id   = state.get("patient_id", "unknown")
    request_type = state.get("request_type", "")
    next_step    = state.get("next_step", "")

    print(f"   patient_id      : {patient_id}")
    print(f"   request_type    : {request_type}")
    print(f"   next_step       : {next_step}")

    # ── Determine which insights field to read ─────────────────────────
    if request_type == "blood_test_analysis":
        insights = state.get("lab_insights")
        prompt_template = DELIVERY_PROMPT_TEMPLATE
        insights_type = "lab_insights"
    elif request_type == "image_lesion_analysis":
        insights = state.get("vision_insights")
        prompt_template = DELIVERY_PROMPT_SKIN_CARE
        insights_type = "vision_insights"
    else:
        log.warning("deliver_node: unknown request_type=%s", request_type)
        insights = None
        prompt_template = DELIVERY_PROMPT_TEMPLATE
        insights_type = "unknown"

    print(f"   {insights_type:15} : {len(insights or '')} chars")

    if not insights:
        log.warning("deliver_node: %s is empty — analyst may have failed", insights_type)
        print(f"   ⚠️  {insights_type} is empty — check analyst output")
        insights = f"No analysis results were returned by the specialist agent ({insights_type})."

    # ── Format delivery prompt ─────────────────────────────────────────
    if request_type == "blood_test_analysis":
        user_prompt = prompt_template.format(
            patient_id=patient_id,
            lab_insights=insights,
        )
    elif request_type == "image_lesion_analysis":
        user_prompt = prompt_template.format(
            patient_id=patient_id,
            vision_insights=insights,
        )
    else:
        user_prompt = f"Patient {patient_id}: {insights}"

    # ── LLM call WITH system prompt ────────────────────────────────────
    print("   🤖 Calling LLM to reshape clinical output → patient message …")
    log.debug("deliver_node: invoking LLM for request_type=%s", request_type)

    response = llm.invoke([
        SystemMessage(content=MANAGER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    final_report = response.content

    print(f"   ✅ final_report  : {len(final_report)} chars")
    print("─" * 50)

    patient_msg = {
        "role":    "assistant",
        "content": final_report,
    }
    trace_msg = {
        "role":    "system",
        "content": (
            f"[Manager → deliver_node] Final patient message generated for {patient_id} "
            f"({request_type}). Length: {len(final_report)} chars."
        ),
    }

    return {
        "final_report": final_report,
        "messages":     [patient_msg, trace_msg],
    }