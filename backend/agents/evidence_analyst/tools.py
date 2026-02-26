from backend.tools.medline_test_rag import create_medline_test_rag
from langchain_core.tools import tool
from typing import Dict, Any, Literal


# ── RAG singleton — initialized once, reused across all tool calls ─────────
_rag_instance = None

def _get_rag():
    global _rag_instance
    if _rag_instance is None:
        print("🔧 [tools] Initializing RAG singleton...")
        _rag_instance = create_medline_test_rag()
        print("✅ [tools] RAG singleton ready")
    return _rag_instance


@tool
def search_medical_knowledge(query: str) -> dict:
    """
    Search medical literature for causes, conditions, and treatments.

    Uses RAG system with MedlinePlus database.

    Args:
        query: Medical question (e.g., "causes of low hemoglobin")

    Returns:
        Medical information from trusted sources
    """
    rag = _get_rag()
    results = rag.answer_question(query)
    return {
        "rag_sys_prompt":  results.get("llm_system_prompt", ""),
        "rag_user_prompt": results.get("llm_user_prompt", ""),
        "answer": f"Medical knowledge about '{query}': {results['answer']}"
    }



