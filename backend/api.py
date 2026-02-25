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
# Helpers — step extraction
# ─────────────────────────────────────────────────────────────────────────

def _build_steps_from_state(
    route_result: dict,
    user_prompt:  str,
    final_state:  dict,
) -> list[dict]:
    """
    Build the ordered step list for the API response.

    Every node (manager_node, specialist agents, deliver_node) writes its
    own step(s) directly into state["steps"] during execution via the
    operator.add reducer — so order is always guaranteed by execution order.

    This function simply:
      1. Prepends the SemanticRouter step (runs before the graph)
      2. Appends the pre-built graph steps from state["steps"]

    No message parsing, no regex, no fallbacks needed.
    """
    # ── Step 1: SemanticRouter — always first, runs before the graph ──────
    router_step = {
        "module":   "Semantic Router",
        "prompt":   user_prompt,
        "response": (
            f"proposed_category={route_result.get('category')} | "
            f"score={route_result.get('score', 0):.4f} | "
            f"confidence={route_result.get('confidence')} | "
            f"passed={route_result.get('passed')} | "
            f"all_scores={route_result.get('all_scores', {})}"
        ),
    }

    # ── Steps 2-N: read directly from state, written by each node ────────
    # Guaranteed order: manager_node → specialist → deliver_node
    # Tool calls are included because each node wrote them at execution time.
    graph_steps = final_state.get("steps", [])

    return [router_step] + graph_steps

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
                        "module": "Deliver Node",
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
                "module": "Semantic Router",
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

        # ── Build state ───────────────────────────────────────────────────
        # For blood test analysis the API uses the first available patient's
        # most recent lab results from Supabase as a demonstration dataset.
        # For skin care analysis the demo image is used automatically by
        # run_skin_care_analyst when image_path is None.
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

        elif proposed_category == "image_lesion_analysis":
            # No image upload in the API — run_skin_care_analyst falls back
            # to the bundled demo image automatically when image_path is None.
            initial_state = {
                "request_type":             proposed_category,
                "patient_id":               patient_id,
                "raw_user_input":           user_prompt,
                "router_proposed_category": proposed_category,
                "router_score":             router_score,
                "router_confidence":        router_confidence,
                "lab_result":               None,
                "lab_insights":             None,
                "image_path":               None,   # triggers demo fallback
                "vision_results":           None,
                "vision_insights":          None,
                "evidence_insights":        None,
                "messages":                 [],
                "next_step":                "",
                "final_report":             None,
                "steps":                    [],
            }

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