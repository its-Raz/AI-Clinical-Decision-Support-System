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
    Receive the analyst's `lab_insights`, generate a patient-friendly
    `final_report`, and append it to messages.
    Uses MANAGER_SYSTEM_PROMPT as SystemMessage so the LLM adopts the
    clinical-liaison persona before generating the patient message.
    """
    log.debug("deliver_node() called")
    print("\n" + "─" * 50)
    print("💬 [deliver_node] ENTER")

    patient_id   = state.get("patient_id", "unknown")
    lab_insights = state.get("lab_insights")
    next_step    = state.get("next_step", "")

    print(f"   patient_id      : {patient_id}")
    print(f"   next_step       : {next_step}")
    print(f"   lab_insights    : {len(lab_insights or '')} chars")

    if not lab_insights:
        log.warning("deliver_node: lab_insights is empty — analyst may have failed")
        print("   ⚠️  lab_insights is empty — check analyst output")
        lab_insights = "No analysis results were returned by the specialist agent."

    # ── Format delivery prompt ─────────────────────────────────────────
    user_prompt = DELIVERY_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        lab_insights=lab_insights,
    )

    # ── LLM call WITH system prompt ────────────────────────────────────
    print("   🤖 Calling LLM with SystemMessage + HumanMessage …")
    log.debug("deliver_node: invoking LLM")

    response = llm.invoke([
        SystemMessage(content=MANAGER_SYSTEM_PROMPT),   # ← role/persona
        HumanMessage(content=user_prompt),              # ← task
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
            f"[Manager → deliver_node] Final report generated for {patient_id}. "
            f"({len(final_report)} chars)"
        ),
    }

    return {
        "final_report": final_report,
        "messages":     [patient_msg, trace_msg],
    }