"""
backend/main.py — Backend orchestration entry point.

Provides the public API consumed by the frontend (app.py).
All routing, state building, and pipeline execution live here.

The router and ManagerAgent are initialised ONCE at module import time,
so the first user request is never slowed down by cold-start initialisation.

Public API:
    initialize()                    → call once at app startup
    route_request(text)             → run semantic router, return routing metadata
    build_blood_test_state(...)     → build AgentState for blood test analysis
    build_lesion_state(...)         → build AgentState for skin lesion analysis
    build_evidence_state(...)       → build AgentState for evidence/Q&A
    execute_pipeline(state)         → run the full agent graph, return final state
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Module-level singletons
# Initialised once when this module is first imported.
# ─────────────────────────────────────────────────────────────────────────

_router = None   # SemanticRouter RouteIndex
_system = None   # ManagerAgent


def initialize() -> None:
    """
    Eagerly build the semantic router index and the ManagerAgent graph.

    Call this once at application startup (before any user interaction).
    Subsequent calls are no-ops — singletons are already built.

    Why eager initialisation matters:
    - The router builds centroids from 24 embedding API calls.
    - The ManagerAgent compiles the LangGraph StateGraph and loads config.
    - If either is built lazily (inside an event handler), the first user
      request pays the full cold-start cost while Streamlit is mid-execution,
      which can cause spinner/cache race conditions.
    """
    global _router, _system

    if _router is None:
        print("🚀 [backend/main.py] Initialising semantic router …")
        from backend.semantic_routing.semantic_router import get_router
        _router = get_router()
        print("   ✅ Semantic router ready.")

    if _system is None:
        print("🚀 [backend/main.py] Initialising ManagerAgent …")
        from backend.agents.graph import build_system
        _system = build_system()
        print("   ✅ ManagerAgent ready.")


def _get_router():
    """Return the router singleton, initialising if necessary."""
    if _router is None:
        initialize()
    return _router


def _get_system():
    """Return the system singleton, initialising if necessary."""
    if _system is None:
        initialize()
    return _system


# ─────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────

def route_request(user_text: str) -> dict:
    """
    Run the semantic router against the user's free-text input.

    Returns:
    {
        "category":   str,    # best route or "unmatched"
        "score":      float,  # cosine similarity
        "all_scores": dict,
        "passed":     bool,   # False → spam/garbage, reject without LLM
        "confidence": str,    # "high" | "medium" | "spam"
    }
    """
    router = _get_router()
    return router.route(user_text)


# ─────────────────────────────────────────────────────────────────────────
# State builders
# Each function returns a fully-populated AgentState dict.
# The `request_type` field is intentionally left blank ("") — the
# Manager / Judge node writes the final accepted category.
# ─────────────────────────────────────────────────────────────────────────

def _base_state(
    user_text: str,
    proposed_category: str,
    router_score: float,
    router_confidence: str,
    patient_id: str,
) -> dict:
    """
    Internal helper: build a base AgentState with all required fields.
    Specialist fields (lab_result, image_path, etc.) default to None
    and are filled in by the specific builder functions below.
    """
    return {
        # ── Semantic router metadata ───────────────────────────────────
        "raw_user_input":           user_text,
        "router_proposed_category": proposed_category,
        "router_score":             router_score,
        "router_confidence":        router_confidence,

        # ── Classification — written by Judge node ─────────────────────
        "request_type":             "",

        # ── Patient ────────────────────────────────────────────────────
        "patient_id":               patient_id,

        # ── Specialist payloads (all None until filled) ────────────────
        "lab_result":               None,
        "lab_insights":             None,
        "image_path":               None,
        "vision_results":           None,
        "vision_insights":          None,
        "evidence_insights":        None,

        # ── Graph internals ────────────────────────────────────────────
        "messages":                 [],
        "next_step":                "",
        "final_report":             None,
    }


def build_blood_test_state(
    user_text: str,
    proposed_category: str,
    router_score: float,
    router_confidence: str,
    patient_id: str,
    lab_result: list,
) -> dict:
    """
    Build an AgentState for a blood test analysis request.

    Args:
        user_text:          Original message from the user.
        proposed_category:  Category suggested by the semantic router.
        router_score:       Cosine similarity score from the router.
        router_confidence:  "high" | "medium" | "spam"
        patient_id:         Patient identifier (e.g. "P001").
        lab_result:         List of lab metric dicts from Supabase.
    """
    state = _base_state(
        user_text=user_text,
        proposed_category=proposed_category,
        router_score=router_score,
        router_confidence=router_confidence,
        patient_id=patient_id,
    )
    state["lab_result"] = lab_result
    return state


def build_lesion_state(
    patient_id: str,
    image_path: str,
) -> dict:
    """
    Build an AgentState for a skin lesion analysis request.

    The image intent is already confirmed by the user uploading a file,
    so router metadata is hardcoded as maximum confidence — the Judge
    will accept this trivially.

    Args:
        patient_id:  Patient identifier.
        image_path:  Absolute path to the saved temp image file.
    """
    state = _base_state(
        user_text="Analyze my skin lesion image",
        proposed_category="image_lesion_analysis",
        router_score=1.0,
        router_confidence="high",
        patient_id=patient_id,
    )
    state["image_path"] = image_path
    return state


def build_evidence_state(
    user_text: str,
    proposed_category: str,
    router_score: float,
    router_confidence: str,
    patient_id: str,
) -> dict:
    """
    Build an AgentState for an evidence / medical Q&A request.

    Args:
        user_text:          Original message from the user.
        proposed_category:  Category suggested by the semantic router.
        router_score:       Cosine similarity score from the router.
        router_confidence:  "high" | "medium" | "spam"
        patient_id:         Patient identifier.
    """
    return _base_state(
        user_text=user_text,
        proposed_category=proposed_category,
        router_score=router_score,
        router_confidence=router_confidence,
        patient_id=patient_id,
    )


# ─────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────

def execute_pipeline(initial_state: dict) -> dict:
    """
    Run the full ManagerAgent graph against an initial AgentState.

    This is the single entry point for all pipeline execution.
    The frontend should never call system.run() directly.

    Args:
        initial_state:  A fully-populated AgentState dict built by one
                        of the build_*_state() helpers above.

    Returns:
        The final AgentState after all nodes have executed, containing
        `final_report`, `messages`, and all specialist outputs.

    Raises:
        Any exception from the graph is re-raised so the frontend can
        display it clearly — we do not swallow errors here.
    """
    system = _get_system()
    log.info(
        "execute_pipeline: patient=%s proposed=%s score=%.4f",
        initial_state.get("patient_id"),
        initial_state.get("router_proposed_category"),
        initial_state.get("router_score", 0.0),
    )
    return system.run(initial_state)


# ─────────────────────────────────────────────────────────────────────────
# Convenience re-exports
# ─────────────────────────────────────────────────────────────────────────

# analyze_existing_test is still used by the frontend to fetch lab data
# from Supabase — we re-export it here so app.py only imports from main.py
from backend.agents.graph import analyze_existing_test   # noqa: E402

__all__ = [
    "initialize",
    "route_request",
    "build_blood_test_state",
    "build_lesion_state",
    "build_evidence_state",
    "execute_pipeline",
    "analyze_existing_test",
]