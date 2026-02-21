"""
backend/api.py — FastAPI HTTP API for the Autonomous Clinical System.

Endpoints:
    GET  /api/team_info           → team metadata
    GET  /api/agent_info          → agent description, prompts, examples
    GET  /api/model_architecture  → PNG architecture diagram
    POST /api/execute             → run the full pipeline, return response + steps

Run with:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

Install dependencies if needed:
    pip install fastapi uvicorn matplotlib
"""

import sys
import os
import re
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from backend.main import (
    initialize,
    route_request,
    build_blood_test_state,
    build_evidence_state,
    execute_pipeline,
)
from backend.supabase.supabase_client import get_patients_summary

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous Clinical System API",
    description="Multi-agent clinical AI: semantic routing, LLM judge, specialist agents.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """
    Build the semantic router index and ManagerAgent graph once at server
    startup so the first POST /api/execute request is not slowed down.
    """
    print("🚀 [api.py] Running startup initialisation …")
    initialize()
    print("✅ [api.py] Ready.")


# ─────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    prompt: str


# ─────────────────────────────────────────────────────────────────────────
# Helpers — step extraction
# ─────────────────────────────────────────────────────────────────────────

# Mapping of bracket prefixes found in state messages → canonical module names.
# These must match the architecture diagram exactly.
_MODULE_MAP = {
    "[judge]":                  "ManagerAgent/Judge",
    "[manager → deliver_node]": "DeliverNode",
    "[manager]":                "ManagerAgent/Judge",
    "[blood_test_analyst]":     "BloodTestAnalyst",
    "[skin_care_analyst]":      "SkinCareAnalyst",
    "[evidence_analyst]":       "EvidenceAnalyst",
    "[deliver_node]":           "DeliverNode",
}


def _detect_module(content: str) -> str | None:
    """
    Detect which module produced a trace message by matching bracketed prefixes.
    Returns the canonical module name or None.
    """
    lower = content.lower()
    for prefix, name in _MODULE_MAP.items():
        if lower.startswith(prefix):
            return name
    return None


def _parse_judge_message(content: str) -> tuple[dict, dict]:
    """
    Extract structured prompt/response fields from the Judge trace message.

    Format written by manager_node:
      [Judge] Patient P001 | Router proposed: 'blood_test_analysis' (0.6632, high) |
      Judge accepted: 'blood_test_analysis' | Overridden: False | Reason: …
    """
    prompt_fields   = {}
    response_fields = {}

    m = re.search(r"Router proposed: '([^']+)' \(([\d.]+), (\w+)\)", content)
    if m:
        prompt_fields["router_proposed_category"] = m.group(1)
        prompt_fields["router_score"]             = float(m.group(2))
        prompt_fields["router_confidence"]        = m.group(3)

    m = re.search(r"Judge accepted: '([^']+)'", content)
    if m:
        response_fields["accepted_category"] = m.group(1)

    m = re.search(r"Overridden: (True|False)", content)
    if m:
        response_fields["overridden"] = m.group(1) == "True"

    m = re.search(r"Reason: (.+)$", content)
    if m:
        response_fields["reasoning"] = m.group(1).strip()

    return prompt_fields, response_fields


def _parse_deliver_message(content: str, final_report: str | None) -> tuple[dict, dict]:
    """
    Extract structured fields from the deliver_node trace message.

    Format:
      [Manager → deliver_node] Final patient message generated for P001
      (blood_test_analysis). Length: 432 chars.
    """
    prompt_fields   = {}
    response_fields = {}

    m = re.search(r"\(([^)]+)\)\. Length: (\d+) chars", content)
    if m:
        prompt_fields["request_type"]    = m.group(1)
        response_fields["report_length"] = int(m.group(2))

    if final_report:
        response_fields["final_report"] = (
            final_report[:500] + "…" if len(final_report) > 500 else final_report
        )

    return prompt_fields, response_fields


def _build_steps_from_state(
    route_result: dict,
    user_prompt:  str,
    final_state:  dict,
) -> list[dict]:
    """
    Construct the ordered list of step objects from the router result
    and the messages written to state by each graph node.

    Step 1 — SemanticRouter       (always present — captured before graph runs)
    Step 2 — ManagerAgent/Judge   (parsed from [Judge] trace message)
    Step 3 — Specialist agent     (BloodTestAnalyst / SkinCareAnalyst / EvidenceAnalyst)
    Step 4 — DeliverNode          (parsed from deliver trace message)
    """
    steps: list[dict] = []

    # ── Step 1: SemanticRouter ────────────────────────────────────────────
    steps.append({
        "module": "SemanticRouter",
        "prompt": {
            "text":            user_prompt,
            "method":          "cosine_similarity",
            "embedding_model": "RPRTHPB-text-embedding-3-small",
        },
        "response": {
            "proposed_category": route_result.get("category"),
            "score":             route_result.get("score"),
            "all_scores":        route_result.get("all_scores", {}),
            "confidence":        route_result.get("confidence"),
            "passed":            route_result.get("passed"),
        },
    })

    # ── Steps 2-4: parse graph messages ──────────────────────────────────
    messages     = final_state.get("messages", [])
    final_report = final_state.get("final_report")
    request_type = final_state.get("request_type", "")

    specialist_module = {
        "blood_test_analysis":   "BloodTestAnalyst",
        "image_lesion_analysis": "SkinCareAnalyst",
        "evidence_analyst":      "EvidenceAnalyst",
    }.get(request_type, "SpecialistAgent")

    judge_added      = False
    specialist_added = False
    deliver_added    = False

    for msg in messages:
        if msg.get("role") != "system":
            continue

        content = msg.get("content", "")
        module  = _detect_module(content)

        if module == "ManagerAgent/Judge" and not judge_added:
            p, r = _parse_judge_message(content)
            p["user_input"] = user_prompt
            steps.append({"module": module, "prompt": p, "response": r})
            judge_added = True

        elif module == specialist_module and not specialist_added:
            insights_key = {
                "BloodTestAnalyst": "lab_insights",
                "SkinCareAnalyst":  "vision_insights",
                "EvidenceAnalyst":  "evidence_insights",
            }.get(module, "insights")

            insights = final_state.get(insights_key, "") or ""
            steps.append({
                "module": module,
                "prompt": {
                    "patient_id":   final_state.get("patient_id"),
                    "request_type": request_type,
                    "trace":        content[:300],
                },
                "response": {
                    "insights_length":  len(insights),
                    "insights_preview": insights[:400] + "…" if len(insights) > 400 else insights,
                },
            })
            specialist_added = True

        elif module == "DeliverNode" and not deliver_added:
            p, r = _parse_deliver_message(content, final_report)
            steps.append({"module": module, "prompt": p, "response": r})
            deliver_added = True

    # ── Fallback: specialist step not found in messages ───────────────────
    if not specialist_added and judge_added:
        insights_key = {
            "blood_test_analysis":   "lab_insights",
            "image_lesion_analysis": "vision_insights",
            "evidence_analyst":      "evidence_insights",
        }.get(request_type, "")

        insights = (final_state.get(insights_key, "") or "") if insights_key else ""
        steps.append({
            "module": specialist_module,
            "prompt": {
                "patient_id":   final_state.get("patient_id"),
                "request_type": request_type,
            },
            "response": {
                "insights_length":  len(insights),
                "insights_preview": insights[:400] + "…" if len(insights) > 400 else insights,
            },
        })

    # ── Fallback: DeliverNode from final_report ───────────────────────────
    if not deliver_added and final_report:
        steps.append({
            "module": "DeliverNode",
            "prompt": {"request_type": request_type},
            "response": {
                "report_length": len(final_report),
                "final_report":  final_report[:500] + "…" if len(final_report) > 500 else final_report,
            },
        })

    return steps


# ─────────────────────────────────────────────────────────────────────────
# A)  GET /api/team_info
# ─────────────────────────────────────────────────────────────────────────

@app.get("/api/team_info")
async def team_info():
    """Return student details and team metadata."""
    return JSONResponse(content={
        "group_batch_order_number": "FILL_IN_BATCH_ORDER",   # ← e.g. "2_5"
        "team_name": "FILL_IN_TEAM_NAME",                    # ← your team name
        "students": [
            {"name": "FILL_IN_NAME", "email": "FILL_IN_EMAIL"},
            {"name": "FILL_IN_NAME", "email": "FILL_IN_EMAIL"},
            {"name": "FILL_IN_NAME", "email": "FILL_IN_EMAIL"},
        ],
    })


# ─────────────────────────────────────────────────────────────────────────
# B)  GET /api/agent_info
# ─────────────────────────────────────────────────────────────────────────

@app.get("/api/agent_info")
async def agent_info():
    """
    Return agent description, purpose, prompt template, and worked examples.

    ⚠️  Replace the full_response and steps values in prompt_examples with
    REAL outputs from your system. Run each prompt through POST /api/execute
    and paste the actual response and steps back here.
    """
    return JSONResponse(content={
        "description": (
            "The Autonomous Clinical System is a multi-agent AI platform that "
            "processes natural-language medical requests. It uses a custom "
            "Semantic Router (OpenAI embeddings + cosine similarity) for fast "
            "intent detection, followed by an LLM-based ManagerAgent/Judge that "
            "accepts or overrides the routing decision using the judge_decision "
            "tool. The accepted category is dispatched to a specialist agent "
            "(BloodTestAnalyst, SkinCareAnalyst, or EvidenceAnalyst), and the "
            "result is reshaped into a patient-friendly report by the DeliverNode."
        ),
        "purpose": (
            "To triage and respond to patient medical queries — including blood "
            "test interpretation, skin lesion screening, and general medical "
            "Q&A — in a safe, empathetic, and clinically grounded way."
        ),
        "prompt_template": {
            "template": (
                "Submit a plain-text medical question or request. Examples:\n"
                "  - Blood test: 'My glucose came back at 112, is that bad?'\n"
                "  - Medical Q&A: 'What are the early signs of kidney disease?'\n"
                "  - Skin concern: 'I have a mole that changed colour recently'\n\n"
                "The system automatically classifies your intent and routes it "
                "to the appropriate specialist. No special syntax required."
            ),
        },
        "prompt_examples": [
            {
                # ── Example 1: Blood test ─────────────────────────────────
                # ⚠️  Replace full_response and steps with real output from
                #     POST /api/execute  {"prompt": "..."}
                "prompt": "My glucose came back at 112 mg/dL and my hemoglobin is slightly low. Can you explain my blood test results?",
                "full_response": "REPLACE_WITH_REAL_RESPONSE_FROM_POST_/api/execute",
                "steps": [
                    {
                        "module": "SemanticRouter",
                        "prompt": {
                            "text": "My glucose came back at 112 mg/dL and my hemoglobin is slightly low. Can you explain my blood test results?",
                            "method": "cosine_similarity",
                            "embedding_model": "RPRTHPB-text-embedding-3-small",
                        },
                        "response": {
                            "proposed_category": "blood_test_analysis",
                            "score": 0.0000,         # ← replace with real score
                            "all_scores": {},         # ← replace with real scores
                            "confidence": "high",
                            "passed": True,
                        },
                    },
                    {
                        "module": "ManagerAgent/Judge",
                        "prompt": {
                            "user_input": "My glucose came back at 112 mg/dL and my hemoglobin is slightly low. Can you explain my blood test results?",
                            "router_proposed_category": "blood_test_analysis",
                            "router_score": 0.0000,   # ← replace with real score
                            "router_confidence": "high",
                        },
                        "response": {
                            "accepted_category": "blood_test_analysis",
                            "overridden": False,
                            "reasoning": "REPLACE_WITH_REAL_REASONING",
                        },
                    },
                    {
                        "module": "BloodTestAnalyst",
                        "prompt": {
                            "patient_id": "REPLACE",
                            "request_type": "blood_test_analysis",
                        },
                        "response": {
                            "insights_preview": "REPLACE_WITH_REAL_INSIGHTS_PREVIEW",
                        },
                    },
                    {
                        "module": "DeliverNode",
                        "prompt": {"request_type": "blood_test_analysis"},
                        "response": {
                            "report_length": 0,       # ← replace with real length
                            "final_report": "REPLACE_WITH_REAL_FINAL_REPORT",
                        },
                    },
                ],
            },
            {
                # ── Example 2: Evidence / medical Q&A ────────────────────
                # ⚠️  Replace full_response and steps with real output from
                #     POST /api/execute  {"prompt": "..."}
                "prompt": "What are the early warning signs of kidney disease?",
                "full_response": "REPLACE_WITH_REAL_RESPONSE_FROM_POST_/api/execute",
                "steps": [
                    {
                        "module": "SemanticRouter",
                        "prompt": {
                            "text": "What are the early warning signs of kidney disease?",
                            "method": "cosine_similarity",
                            "embedding_model": "RPRTHPB-text-embedding-3-small",
                        },
                        "response": {
                            "proposed_category": "evidence_analyst",
                            "score": 0.0000,          # ← replace with real score
                            "all_scores": {},          # ← replace with real scores
                            "confidence": "high",
                            "passed": True,
                        },
                    },
                    {
                        "module": "ManagerAgent/Judge",
                        "prompt": {
                            "user_input": "What are the early warning signs of kidney disease?",
                            "router_proposed_category": "evidence_analyst",
                            "router_score": 0.0000,   # ← replace with real score
                            "router_confidence": "high",
                        },
                        "response": {
                            "accepted_category": "evidence_analyst",
                            "overridden": False,
                            "reasoning": "REPLACE_WITH_REAL_REASONING",
                        },
                    },
                    {
                        "module": "EvidenceAnalyst",
                        "prompt": {
                            "patient_id": "API_USER",
                            "request_type": "evidence_analyst",
                        },
                        "response": {
                            "insights_preview": "REPLACE_WITH_REAL_INSIGHTS_PREVIEW",
                        },
                    },
                    {
                        "module": "DeliverNode",
                        "prompt": {"request_type": "evidence_analyst"},
                        "response": {
                            "report_length": 0,       # ← replace with real length
                            "final_report": "REPLACE_WITH_REAL_FINAL_REPORT",
                        },
                    },
                ],
            },
        ],
    })


# ─────────────────────────────────────────────────────────────────────────
# C)  GET /api/model_architecture
# ─────────────────────────────────────────────────────────────────────────

@app.get("/api/model_architecture")
async def model_architecture():
    """
    Return the system architecture as a PNG image.
    Generated in memory by matplotlib — no static file needed.
    """
    try:
        from backend.architecture_diagram import generate_architecture_png
        png_bytes = generate_architecture_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        log.exception("model_architecture: failed to generate PNG")
        raise HTTPException(status_code=500, detail=f"Diagram generation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# D)  POST /api/execute
# ─────────────────────────────────────────────────────────────────────────

@app.post("/api/execute")
async def execute(body: ExecuteRequest):
    """
    Main pipeline entry point.

    Accepts a plain-text medical prompt, runs the full multi-agent pipeline,
    and returns the patient-friendly response + a structured step trace.

    Pipeline:
        SemanticRouter → ManagerAgent/Judge → Specialist → DeliverNode
    """
    user_prompt = body.prompt.strip()

    if not user_prompt:
        return JSONResponse(content={
            "status":   "error",
            "error":    "Prompt must not be empty.",
            "response": None,
            "steps":    [],
        }, status_code=400)

    steps: list[dict] = []

    try:
        # ── Step 1: Semantic routing ──────────────────────────────────────
        route_result      = route_request(user_prompt)
        proposed_category = route_result["category"]
        router_score      = route_result["score"]
        router_confidence = route_result["confidence"]
        passed            = route_result["passed"]

        # ── Spam gate ─────────────────────────────────────────────────────
        if not passed:
            steps.append({
                "module": "SemanticRouter",
                "prompt": {
                    "text":   user_prompt,
                    "method": "cosine_similarity",
                },
                "response": {
                    "proposed_category": "unmatched",
                    "score":             router_score,
                    "confidence":        "spam",
                    "passed":            False,
                },
            })
            return JSONResponse(content={
                "status":   "ok",
                "error":    None,
                "response": (
                    "I'm sorry, I can only assist with medical questions. "
                    "Please describe a health concern, lab result, or symptom."
                ),
                "steps": steps,
            })

        # ── image_lesion_analysis — cannot process without an image ───────
        if proposed_category == "image_lesion_analysis":
            steps.append({
                "module": "SemanticRouter",
                "prompt": {"text": user_prompt, "method": "cosine_similarity"},
                "response": {
                    "proposed_category": proposed_category,
                    "score":             router_score,
                    "confidence":        router_confidence,
                    "passed":            True,
                },
            })
            return JSONResponse(content={
                "status":   "ok",
                "error":    None,
                "response": (
                    "Skin lesion analysis requires an image upload. "
                    "Please use the Clinical Portal web interface to upload "
                    "a photo of the lesion for analysis."
                ),
                "steps": steps,
            })

        # ── Build state ───────────────────────────────────────────────────
        # For blood test analysis the API uses the first available patient's
        # most recent lab results from Supabase as a demonstration dataset.
        patient_id = "API_USER"

        if proposed_category == "blood_test_analysis":
            try:
                from backend.agents.graph import analyze_existing_test

                patients = get_patients_summary()
                if patients:
                    patient_id = patients[0]["id"]
                    lab_state  = analyze_existing_test(patient_id, -1)
                    lab_result = lab_state.get("lab_result", [])
                else:
                    lab_result = []
            except Exception:
                lab_result = []

            initial_state = build_blood_test_state(
                user_text         = user_prompt,
                proposed_category = proposed_category,
                router_score      = router_score,
                router_confidence = router_confidence,
                patient_id        = patient_id,
                lab_result        = lab_result,
            )

        else:  # evidence_analyst
            initial_state = build_evidence_state(
                user_text         = user_prompt,
                proposed_category = proposed_category,
                router_score      = router_score,
                router_confidence = router_confidence,
                patient_id        = patient_id,
            )

        # ── Execute full pipeline ─────────────────────────────────────────
        final_state  = execute_pipeline(initial_state)
        final_report = final_state.get("final_report") or ""

        # ── Build structured steps ────────────────────────────────────────
        steps = _build_steps_from_state(route_result, user_prompt, final_state)

        return JSONResponse(content={
            "status":   "ok",
            "error":    None,
            "response": final_report,
            "steps":    steps,
        })

    except Exception as exc:
        log.exception("execute: pipeline failed")
        return JSONResponse(
            status_code=500,
            content={
                "status":   "error",
                "error":    str(exc),
                "response": None,
                "steps":    steps,  # return whatever steps were captured so far
            },
        )