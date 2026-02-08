import os
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import yaml
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Load environment variables
load_dotenv()




class KnowledgeBaseConfig(BaseModel):
    """Configuration for a single knowledge base."""

    name: str = Field(..., description="Display name of the knowledge base")
    description: str = Field(..., description="Description of the knowledge base")
    bm25_index_path: str = Field(..., description="Path to BM25 index file")
    pinecone_index_name: str = Field(..., description="Environment variable name for Pinecone index")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dimensions: int = Field(..., description="Embedding dimensions")
    bm25_k: int = Field(default=20, description="Number of results from BM25")
    vector_k: int = Field(default=20, description="Number of results from vector search")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for BM25 results")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector results")
    final_k: int = Field(default=10, description="Final number of results after fusion")

    @field_validator('bm25_index_path')
    def validate_bm25_path(cls, v):
        """Ensure BM25 index path exists."""
        path = Path(v)
        if not path.is_absolute():
            # Resolve relative to this file's directory
            script_dir = Path(__file__).parent
            path = (script_dir / path).resolve()

        if not path.exists():
            raise ValueError(f"BM25 index not found at: {path}")
        return str(path)

    @field_validator('pinecone_index_name')
    def validate_pinecone_index(cls, v):
        """Get actual index name from environment variable."""
        index_name = os.getenv(v)
        if not index_name:
            raise ValueError(f"Environment variable '{v}' not found in .env file")
        return index_name


class RRFConfig(BaseModel):
    """Configuration for Reciprocal Rank Fusion."""

    c: int = Field(default=60, description="Constant for RRF formula: 1 / (rank + c)")


class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings API."""

    base_url: str = Field(..., description="API base URL")
    api_key_env: str = Field(..., description="Environment variable name for API key")

    @field_validator('api_key_env')
    def validate_api_key(cls, v):
        """Ensure API key exists in environment."""
        api_key = os.getenv(v)
        if not api_key:
            raise ValueError(f"Environment variable '{v}' not found in .env file")
        return v

class RAGConfig(BaseModel):
    """Complete RAG system configuration."""

    knowledge_bases: Dict[str, KnowledgeBaseConfig]
    rrf: RRFConfig
    embeddings: EmbeddingsConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)


class MedlineTestRAG:
    """
    Hybrid RAG system combining BM25 (keyword) and Vector (semantic) search
    with Reciprocal Rank Fusion.
    """

    def __init__(
            self,
            kb_name: Literal["medline_test"],
            config_path: Optional[str] = None
    ):
        """
        Initialize RAG system for a specific knowledge base.

        Args:
            kb_name: Name of the knowledge base to use
            config_path: Path to YAML config file (default: ../config/rag_config.yaml)
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "tools" / "config" / "medline_test_rag_config.yaml"

        self.config = RAGConfig.from_yaml(config_path)

        # Get knowledge base config
        if kb_name not in self.config.knowledge_bases:
            raise ValueError(
                f"Knowledge base '{kb_name}' not found in config. "
                f"Available: {list(self.config.knowledge_bases.keys())}"
            )

        self.kb_config = self.config.knowledge_bases[kb_name]
        self.kb_name = kb_name

        print(f"Initializing RAG for: {self.kb_config.name}")

        # Initialize components
        self._initialize_embeddings()
        self._initialize_bm25_retriever()
        self._initialize_vector_retriever()
        self._initialize_ensemble_retriever()

        print(f"RAG system ready!")

    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings via LLMOD."""
        api_key = os.getenv(self.config.embeddings.api_key_env)

        self.embeddings = OpenAIEmbeddings(
            model=self.kb_config.embedding_model,
            openai_api_key=api_key,
            base_url=self.config.embeddings.base_url,
            dimensions=self.kb_config.embedding_dimensions
        )

        print(f"Embeddings initialized: {self.kb_config.embedding_model}")

    def _initialize_bm25_retriever(self):
        """Load BM25 index from disk."""
        import pickle

        bm25_path = Path(self.kb_config.bm25_index_path)

        print(f"Loading BM25 index from: {bm25_path}")

        with open(bm25_path, 'rb') as f:
            self.bm25_retriever = pickle.load(f)

        # Set k for BM25
        self.bm25_retriever.k = self.kb_config.bm25_k

        print(f"BM25 retriever loaded ({len(self.bm25_retriever.docs)} documents)")

    def _initialize_vector_retriever(self):
        """Initialize Pinecone vector store retriever."""
        pinecone_api_key = os.getenv("PINECONE_API_KEY")

        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")

        print(f"Connecting to Pinecone index: {self.kb_config.pinecone_index_name}")

        # Initialize Pinecone vector store
        vectorstore = PineconeVectorStore(
            index_name=self.kb_config.pinecone_index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key
        )

        # Create retriever
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.kb_config.vector_k}
        )

        print(f"✓ Vector retriever initialized")

    def _initialize_ensemble_retriever(self):
        """Initialize EnsembleRetriever with RRF."""
        print(f"Creating EnsembleRetriever with weights: "
              f"BM25={self.kb_config.bm25_weight}, "
              f"Vector={self.kb_config.vector_weight}")

        # EnsembleRetriever automatically uses RRF (Reciprocal Rank Fusion)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.kb_config.bm25_weight, self.kb_config.vector_weight],
            c=self.config.rrf.c  # RRF constant (default is 60)
        )

        print(f"✓ EnsembleRetriever created (using RRF)")

    def query(
            self,
            query: str,
            k: Optional[int] = None
    ) -> List[Document]:
        """
        Query the knowledge base using hybrid search with RRF.

```
        """
        k = k or self.kb_config.k

        print(f"\nQuerying: '{query}'")
        print(f"  - Using EnsembleRetriever with RRF")
        print(f"  - Weights: BM25={self.kb_config.bm25_weight}, Vector={self.kb_config.vector_weight}")

        # Use EnsembleRetriever (automatically does RRF)
        results = self.ensemble_retriever.invoke(query)

        # Return top k results
        results = results[:k]

        print(f"✓ Returning top {len(results)} results")

        return results

    def get_full_content(self, doc: Document) -> str:
        """
        Get full content from a document.

        If BM25 indexing used text_extraction_fn, the full content
        is stored in metadata['original_content'].
        """
        return doc.metadata.get('original_content', doc.page_content)

    def query_bm25_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Query using BM25 only (keyword search)."""
        k = k or self.kb_config.k
        original_k = self.bm25_retriever.k
        self.bm25_retriever.k = k

        results = self.bm25_retriever.invoke(query)

        self.bm25_retriever.k = original_k
        return results

    def query_vector_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Query using vector search only (semantic search)."""
        k = k or self.kb_config.k

        # Temporarily update search_kwargs
        original_k = self.vector_retriever.search_kwargs.get('k', 10)
        self.vector_retriever.search_kwargs['k'] = k

        results = self.vector_retriever.invoke(query)

        # Restore original k
        self.vector_retriever.search_kwargs['k'] = original_k
        return results


# Convenience functions for each knowledge base

def create_medline_test_rag(config_path: Optional[str] = None) -> MedlineTestRAG:
    """Create RAG for MedlinePlus knowledge base."""
    return MedlineTestRAG(kb_name="medline_test", config_path=config_path)


def example_basic_usage():
    """Basic usage with EnsembleRetriever."""

    # Initialize RAG (automatically uses EnsembleRetriever with RRF)
    rag = create_medline_test_rag()

    # Query using hybrid search (BM25 + Vector with RRF)
    results = rag.query(
        query="Acetaminophen Level",
        k=10
    )

    print(f"\nTop {len(results)} results (ranked by RRF):\n")

    for i, doc in enumerate(results, 1):
        full_content = rag.get_full_content(doc)
        print(f"{i}. {doc.metadata.get('Doc_Title')} - {doc.metadata.get('Sec_Title')}")
        print(f"   {full_content[:200]}...")
        print()

if __name__ == "__main__":
    example_basic_usage()