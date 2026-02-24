"""
Manager node logic.

Two nodes:
  manager_node  – inspects state, logs routing decision, appends a trace message
  deliver_node  – takes lab_insights from the analyst and generates the
                  patient-facing final_report via LLM, using the system prompt
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from backend.agents.manager.prompts import MANAGER_SYSTEM_PROMPT, DELIVERY_PROMPT_TEMPLATE

log = logging.getLogger(__name__)


def manager_node(state: dict, llm) -> dict:
    """
    Acts as the Judge for the semantic router's proposed classification.
    Forces a structured decision via tool calling — no string parsing needed.

    Reads:  raw_user_input, router_proposed_category, router_score, router_confidence
    Writes: request_type (final accepted category), next_step, messages
    """
    from .prompts import JUDGE_PROMPT, MANAGER_SYSTEM_PROMPT
    from .tools import judge_decision
    from langchain_core.messages import HumanMessage, SystemMessage

    log.debug("manager_node() / Judge called")
    print("\n" + "─" * 50)
    print("⚖️  [manager_node / Judge] ENTER")

    patient_id        = state.get("patient_id", "unknown")
    user_input        = state.get("raw_user_input", "")
    proposed_category = state.get("router_proposed_category", "unsupported")
    router_score      = state.get("router_score", 0.0)
    confidence        = state.get("router_confidence", "medium")

    print(f"   patient_id        : {patient_id}")
    print(f"   proposed_category : {proposed_category}")
    print(f"   router_score      : {router_score:.4f}")
    print(f"   confidence        : {confidence}")

    # ── Build Judge prompt ────────────────────────────────────────────
    prompt = JUDGE_PROMPT.format(
        user_input=user_input,
        proposed_category=proposed_category,
        router_score=router_score,
        confidence=confidence,
    )
    full_prompt_text = f"[SYSTEM]\n{MANAGER_SYSTEM_PROMPT}\n\n[USER]\n{prompt}"
    # ── Force tool call — LLM cannot respond in plain text ───────────
    llm_judge = llm.bind_tools([judge_decision])

    print("   🤖 Calling Judge LLM …")
    response = llm_judge.invoke([
        SystemMessage(content=MANAGER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    # ── Extract structured args from tool call ────────────────────────
    tool_calls = response.tool_calls

    if tool_calls:
        args              = tool_calls[0]["args"]
        accepted_category = args["accepted_category"]
        reasoning         = args["reasoning"]
        overridden        = args.get("overridden", False)
    else:
        # Should never happen with tool_choice enforced — fail safely
        log.error("Judge LLM did not call judge_decision — falling back to router proposal")
        accepted_category = proposed_category
        reasoning         = "Tool call missing — router proposal used as fallback."
        overridden        = False

    print(f"   ✅ accepted_category : {accepted_category}")
    print(f"   📝 reasoning         : {reasoning}")
    print(f"   🔀 overridden        : {overridden}")

    # ── Map accepted category → next graph node ───────────────────────
    route_map = {
        "blood_test_analysis":   "blood_test_analyst",
        "image_lesion_analysis": "skin_care_analyst",
        "evidence_analyst":      "evidence_analyst",
        "unsupported":           "deliver",
    }
    next_step = route_map.get(accepted_category, "deliver")

    print(f"   ↳ next_step : {next_step}")
    print("─" * 50)

    # ── Trace message for audit log ───────────────────────────────────
    trace_msg = {
        "role": "system",
        "content": (
            f"[Judge] Patient {patient_id} | "
            f"Router proposed: '{proposed_category}' ({router_score:.4f}, {confidence}) | "
            f"Judge accepted: '{accepted_category}' | "
            f"Overridden: {overridden} | "
            f"Reason: {reasoning}"
        ),
    }

    # ── Step 1: the LLM call ──────────────────────────────────────────
    judge_step = {
        "module": "ManagerAgent/Judge",
        "prompt": full_prompt_text,
        "response": response.content or f"[tool call triggered] {args}",
    }

    # ── Step 2: the tool call result ──────────────────────────────────
    tool_step = {
        "module": "judge_decision (tool)",
        "prompt": (
            f"accepted_category: {accepted_category}\n"
            f"reasoning: {reasoning}\n"
            f"overridden: {overridden}"
        ),
        "response": (
            f"Routing to: {next_step} | "
            f"Accepted: {accepted_category} | "
            f"Overridden: {overridden}"
        ),
    }

    return {
        "request_type": accepted_category,
        "next_step": next_step,
        "messages": [trace_msg],
        "steps": [judge_step, tool_step],
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


    elif request_type == "evidence_analyst":
        insights = state.get("evidence_insights")
        from .prompts import DELIVERY_PROMPT_EVIDENCE  # ייבוא הפרומפט החדש
        prompt_template = DELIVERY_PROMPT_EVIDENCE
        insights_type = "evidence_insights"
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
    elif request_type == "evidence_analyst":
        user_prompt = prompt_template.format(
            patient_id=patient_id,
            evidence_insights=insights,
        )
    else:
        user_prompt = f"Patient {patient_id}: {insights}"
    full_prompt_text = f"[SYSTEM]\n{MANAGER_SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}"

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

    deliver_step = {
        "module": "DeliverNode",
        "prompt": full_prompt_text,
        "response": final_report,
    }

    return {
        "final_report": final_report,
        "messages": [patient_msg, trace_msg],
        "steps": [deliver_step],
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