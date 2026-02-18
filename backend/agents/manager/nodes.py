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

        # Build raw results table
        lab_results = state.get("lab_result") or []
        print(f"   lab_result count: {len(lab_results)}")
        raw_results = _format_raw_results_table(lab_results)
        print(f"   raw_results table: {len(raw_results)} chars")
        print(f"   raw_results preview:\n{raw_results[:300]}")

    elif request_type == "image_lesion_analysis":
        insights = state.get("vision_insights")
        prompt_template = DELIVERY_PROMPT_SKIN_CARE
        insights_type = "vision_insights"
        raw_results = None

    else:
        log.warning("deliver_node: unknown request_type=%s", request_type)
        insights = None
        prompt_template = DELIVERY_PROMPT_TEMPLATE
        insights_type = "unknown"
        raw_results = None

    print(f"   {insights_type:15} : {len(insights or '')} chars")

    if not insights:
        log.warning("deliver_node: %s is empty — analyst may have failed", insights_type)
        print(f"   ⚠️  {insights_type} is empty — check analyst output")
        insights = f"No analysis results were returned by the specialist agent ({insights_type})."

    # ── Format delivery prompt ─────────────────────────────────────────
    if request_type == "blood_test_analysis":
        user_prompt = prompt_template.format(
            patient_id=patient_id,
            raw_results=raw_results,
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

    print(f"\n   📤 PROMPT PREVIEW (first 800 chars):")
    print(f"   {user_prompt[:800]}...")

    response = llm.invoke([
        SystemMessage(content=MANAGER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    print(f"\n   📥 LLM RAW RESPONSE:")
    print(f"      type: {type(response)}")
    print(f"      content type: {type(response.content)}")
    print(f"      content length: {len(response.content)}")
    print(f"      content preview: {repr(response.content[:200]) if response.content else 'EMPTY'}")
    print(f"      response_metadata: {response.response_metadata}")

    final_report = response.content

    if not final_report:
        log.error("deliver_node: LLM returned EMPTY content!")
        print("   ❌ LLM returned empty content - check prompt/model configuration")
        final_report = (
            "Error: The system was unable to generate a patient-friendly summary. "
            "Please contact your care team for assistance."
        )

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


def _format_raw_results_table(lab_results: list) -> str:
    """Format lab results into a markdown table for the delivery prompt."""
    if not lab_results:
        return "No lab results available."

    # Reference ranges (simplified - could be made dynamic)
    ref_ranges = {
        "Glucose": "70-100 mg/dL",
        "HbA1c": "<5.7%",
        "Hemoglobin": "12.0-15.5 g/dL (F) / 13.5-17.5 g/dL (M)",
        "Creatinine": "0.6-1.2 mg/dL",
    }

    # Build table
    lines = []
    for result in lab_results:
        test = result.get("test_name", "Unknown")
        value = result.get("value", "?")
        unit = result.get("unit", "")
        flag = result.get("flag", "normal")

        # Map flag to status emoji
        status_map = {
            "normal": "✅ Normal",
            "low": "⬇️ Low",
            "high": "⬆️ High",
            "borderline_low": "⚠️ Borderline Low",
            "borderline_high": "⚠️ Borderline High",
            "critical_low": "🔴 Critical Low",
            "critical_high": "🔴 Critical High",
        }
        status = status_map.get(flag, flag)

        ref_range = ref_ranges.get(test, "See specialist")

        lines.append(f"| {test} | {value} {unit} | {ref_range} | {status} |")

    return "\n".join(lines)