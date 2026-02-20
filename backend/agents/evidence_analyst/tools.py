from backend.tools.medline_test_rag import create_medline_test_rag
from langchain_core.tools import tool
from typing import Dict, Any, Literal



@tool
def search_medical_knowledge(query: str):
    """
        Search medical literature for causes, conditions, and treatments.

        Uses RAG system with MedlinePlus database.

        Args:
            query: Medical question (e.g., "causes of low hemoglobin")

        Returns:
            Medical information from trusted sources
        """
    rag = create_medline_test_rag()
    results = rag.answer_question(query)
    return f"Medical knowledge about '{query}': {results}"



