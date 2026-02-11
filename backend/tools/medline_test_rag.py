import os
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import yaml
from dotenv import load_dotenv
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from prompts import MEDLINE_TEST_SYSTEM_PROMPT, MEDLINE_TEST_QUERY_PROMPT
# Load environment variables
load_dotenv()




class KnowledgeBaseConfig(BaseModel):
    """Configuration for a single knowledge base."""

    name: str = Field(..., description="Display name of the knowledge base")
    description: str = Field(..., description="Description of the knowledge base")
    bm25_index_path: str = Field(..., description="Path to BM25 index file")
    pinecone_index_name: str = Field(..., description="Environment variable name for Pinecone index")
    embedding_model: str = Field(..., description="Embedding model name")
    LLM_model: str = Field(..., description="LLN model name")
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

class RerankerConfig(BaseModel):
    """Configuration for Reranker."""
    use_reranker: bool = Field(default=False, description="Whether to use reranker")
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross encoder model name")
    top_k: int = Field(default=5, description="Number of results to return after reranking")

    @field_validator('use_reranker', mode='before')
    def convert_yes_no(cls, v):
        """Convert 'yes'/'no' strings to boolean."""
        if isinstance(v, str):
            return v.lower() in ('yes', 'true', '1')
        return bool(v)

class LLMConfig(BaseModel):
    """Configuration for LLM"""

    base_url: str = Field(..., description="API base URL")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    max_tokens: int = Field(default=500, description="Number of max_tokens")
    @field_validator('api_key_env')
    def validate_api_key(cls, v):
        """Ensure API key exists in environment."""
        api_key = os.getenv(v)
        if not api_key:
            raise ValueError(f"Environment variable '{v}' not found in .env file")
        return v

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
    reranker: RerankerConfig
    LLM: LLMConfig

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
        self._initialize_LLM()
        self._load_prompts()
        if self.config.reranker.use_reranker:
            self._initialize_reranker()

        if self.kb_config.bm25_weight > 0:
            self._initialize_bm25_retriever()

        if self.kb_config.vector_weight > 0:
            self._initialize_vector_retriever()

        if self.kb_config.bm25_weight > 0 and self.kb_config.vector_weight > 0:
            self._initialize_ensemble_retriever()
        print(f"✓ RAG system ready!")

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
    def _initialize_LLM(self):
        """Initialize OpenAI embeddings via LLMOD."""
        api_key = os.getenv(self.config.LLM.api_key_env)

        self.LLM = ChatOpenAI(
            model=self.kb_config.LLM_model,
            openai_api_key=api_key,
            base_url=self.config.LLM.base_url,
            temperature = 1,
            max_tokens = self.config.LLM.max_tokens

        )

        print(f"LLM initialized: {self.kb_config.LLM_model}")

    def _load_prompts(self):
        """Load prompts for this knowledge base."""
        if self.kb_name == "medline_test":

            self.system_prompt = MEDLINE_TEST_SYSTEM_PROMPT
            self.query_template = MEDLINE_TEST_QUERY_PROMPT
        else:
            raise ValueError(f"No prompts defined for knowledge base: {self.kb_name}")

    def _initialize_reranker(self):
        """Initialize cross-encoder reranker."""
        print(f"Loading reranker model: {self.config.reranker.model_name}")

        self.reranker = CrossEncoder(self.config.reranker.model_name)

        print(f"✓ Reranker initialized: {self.config.reranker.model_name}")
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

    def _deduplicate_results(self, results: List[Document]) -> List[Document]:
        unique_results = []
        seen_signatures = set()

        for doc in results:
            doc_title = str(doc.metadata.get('Doc_Title', '')).strip()
            sec_title = str(doc.metadata.get('Sec_Title', '')).strip()

            raw_idx = doc.metadata.get('Chunk_Index')
            try:
                chunk_index = int(float(raw_idx)) if raw_idx is not None else None
            except (ValueError, TypeError) as e:
                print(f"⚠️ Warning: Invalid Chunk_Index '{raw_idx}' for {doc_title}/{sec_title}")
                chunk_index = None  # Or use a sentinel value like -1

            signature = (doc_title, sec_title, chunk_index)

            if signature not in seen_signatures:
                unique_results.append(doc)
                seen_signatures.add(signature)

        return unique_results

    def _rerank_results(self, query: str, results: List[Document], k: int) -> List[Document]:
        """
        Rerank results using cross-encoder model.

        Args:
            query: Search query
            results: Initial retrieved results
            k: Number of top results to return after reranking

        Returns:
            Reranked list of documents
        """
        if not results:
            return results


        # Prepare query-document pairs
        pairs = [[query, self.get_full_content(doc)] for doc in results]

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Sort by scores (descending)
        scored_results = list(zip(results, scores))

        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        reranked = [doc for doc, score in scored_results[:k]]

        print(f"✓ Reranked {len(results)} -> {len(reranked)} results")

        return reranked

    def query(
            self,
            query: str,
            k: Optional[int] = None
    ) -> List[Document]:
        """
        Query the knowledge base using hybrid search with RRF and optional reranking.

        Args:
            query: Search query string
            k: Number of final results to return (uses config.final_k if None)

        Returns:
            List of Document objects ranked by relevance
        """
        k = k or self.kb_config.final_k

        print(f"\n{'=' * 60}")
        print(f"Query: '{query}'")
        print(f"{'=' * 60}")

        # Determine which retrieval method to use
        if self.kb_config.bm25_weight > 0 and self.kb_config.vector_weight > 0:
            # Use ensemble (hybrid) retrieval with RRF
            print("Using: Hybrid search (BM25 + Vector + RRF)")
            raw_results = self.ensemble_retriever.invoke(query)

        elif self.kb_config.bm25_weight > 0:
            # BM25 only
            print("Using: BM25 only")
            raw_results = self.query_bm25_only(query, k=self.kb_config.bm25_k)

        elif self.kb_config.vector_weight > 0:
            # Vector only
            print("Using: Vector search only")
            raw_results = self.query_vector_only(query, k=self.kb_config.vector_k)

        else:
            raise ValueError("At least one of bm25_weight or vector_weight must be > 0")

        print(f"✓ Retrieved {len(raw_results)} initial results")

        # Deduplicate results
        # If using reranker, get more candidates before reranking

        unique_results = self._deduplicate_results(raw_results)

        print(f"✓ After deduplication: {len(unique_results)} unique results")

        # Apply reranking if enabled
        if self.config.reranker.use_reranker and len(unique_results) > 0:
            final_results = self._rerank_results(query, unique_results, k=k)
        else:
            final_results = unique_results[:k]

        print(f"✓ Returning {len(final_results)} final results")
        print(f"{'=' * 60}\n")

        return final_results
    def get_full_content(self, doc: Document) -> str:
        """
        Get full content from a document.

        If BM25 indexing used text_extraction_fn, the full content
        is stored in metadata['original_content'].
        """
        return doc.metadata.get('original_content', doc.page_content)

    def query_bm25_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Query using BM25 only (keyword search)."""
        k = k or self.kb_config.final_k
        original_k = self.bm25_retriever.k
        self.bm25_retriever.k = k

        results = self.bm25_retriever.invoke(query)

        self.bm25_retriever.k = original_k
        return results

    def query_vector_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Query using vector search only (semantic search)."""
        k = k or self.kb_config.final_k

        # Temporarily update search_kwargs
        original_k = self.vector_retriever.search_kwargs.get('k', 10)
        self.vector_retriever.search_kwargs['k'] = k

        results = self.vector_retriever.invoke(query)

        # Restore original k
        self.vector_retriever.search_kwargs['k'] = original_k
        return results

    def build_augmented_prompt(
            self,
            query: str,
            retrieved_docs: List[Document]
    ) -> str:
        """
        Build augmented prompt by combining query with retrieved context.

        Args:
            query: User's question
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted prompt string with context and query
        """


        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # doc_title = doc.metadata.get('Doc_Title', 'Unknown Document')
            # sec_title = doc.metadata.get('Sec_Title', 'Unknown Section')
            # chunk_index = doc.metadata.get("Chunk_Index","Unknown Chunk Index")
            # splitted = doc.metadata.get("Splitted","Unknown")
            content = self.get_full_content(doc)

            context_parts.append(
                f"[Source {i}]\n{content}"
            )

        context = "\n\n".join(context_parts)

        # Format the query prompt with context and question
        # Assuming the query_template has placeholders for {context} and {question}
        augmented_prompt = self.query_template.format(
            context=context,
            question=query
        )



        return augmented_prompt

    def answer_question(
            self,
            question: str,
            k: Optional[int] = None,
            return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.

        Args:
            question: User's question
            k: Number of documents to retrieve (uses config.final_k if None)
            return_sources: Whether to include source documents in response

        Returns:
            Dictionary containing:
                - answer: Generated answer
                - sources: List of source documents (if return_sources=True)
                - query: Original question
        """
        # Input validation
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        print(f"\n{'*' * 60}")
        print(f"ANSWERING QUESTION")
        print(f"{'*' * 60}")

        # Step 1: Retrieve relevant documents
        print("\n[1/3] Retrieving relevant documents...")
        retrieved_docs = self.query(query=question, k=k)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query": question
            }

        print(f"✓ Retrieved {len(retrieved_docs)} relevant documents")

        # Step 2: Build context and user message
        print("\n[2/3] Building prompts with context...")

        user_message = self.build_augmented_prompt(question, retrieved_docs)
        print("✓ System")
        print(self.system_prompt)
        print("✓ User")
        print(user_message)
        print("✓ Prompts constructed")

        # Step 3: Generate answer using LLM
        print("\n[3/3] Generating answer...")

        try:
            messages = [
                ("system", self.system_prompt),
                ("human", user_message)
            ]
            response = self.LLM.invoke(messages)
            answer = response.content
            print("✓ Answer generated")
            print(answer)
        except Exception as e:
            print(f"✗ Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": retrieved_docs if return_sources else [],
                "query": question
            }

        print(f"{'*' * 60}\n")

        # Build response
        result = {
            "answer": answer,
            "query": question
        }

        if return_sources:
            result["sources"] = retrieved_docs

        return result

# Convenience functions for each knowledge base

def create_medline_test_rag(config_path: Optional[str] = None) -> MedlineTestRAG:
    """Create RAG for MedlinePlus knowledge base."""
    return MedlineTestRAG(kb_name="medline_test", config_path=config_path)


def example_basic_usage():
    """Basic usage with EnsembleRetriever."""

    # Initialize RAG (automatically uses EnsembleRetriever with RRF)
    rag = create_medline_test_rag()
    rag.answer_question("What are the primary risks associated with a 17-hydroxyprogesterone blood test?")
    # # Query using hybrid search (BM25 + Vector with RRF)
    # results = rag.query(
    #     query="Acetaminophen Level"
    # )
    #
    # print(f"\nTop {len(results)} results (ranked by RRF):\n")
    #
    # for i, doc in enumerate(results, 1):
    #     full_content = rag.get_full_content(doc)
    #     print(f"{i}. {doc.metadata.get('Doc_Title')} - {doc.metadata.get('Sec_Title')}")
    #     print(f"   {full_content[:200]}...")
    #     print()

if __name__ == "__main__":
    example_basic_usage()