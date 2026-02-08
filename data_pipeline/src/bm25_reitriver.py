# utils.py

import pickle
import os
from typing import List, Callable, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def build_and_save_bm25_index(
        documents: List[Document],
        save_path: str,
        text_extraction_fn: Optional[Callable[[Document], str]] = None,
        overwrite: bool = False,
) -> BM25Retriever:

    # Validation
    if not documents:
        raise ValueError("Documents list cannot be empty")

    save_path = Path(save_path)

    # Check if file exists
    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"Index file already exists at {save_path}. "
            f"Set overwrite=True to replace it."
        )

    # Create directory if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building BM25 index for {len(documents)} documents...")

    # Prepare documents for indexing
    processed_docs = documents.copy()

    # If custom text extraction is provided, modify documents
    if text_extraction_fn is not None:
        print("Extracting searchable text from documents...")
        processed_docs = []

        for doc in documents:
            # Extract searchable text (e.g., titles)
            search_text = text_extraction_fn(doc)

            # Create new document with:
            # - page_content = searchable text (what BM25 searches on)
            # - metadata = original metadata + original full content
            new_doc = Document(
                page_content=search_text,
                metadata={
                    **doc.metadata,
                    'original_content': doc.page_content  # Store full chunk here
                }
            )
            processed_docs.append(new_doc)

        print(f"✓ Extracted search text from {len(processed_docs)} documents")

    # Create BM25Retriever using LangChain
    print("Building BM25 retriever...")
    retriever = BM25Retriever.from_documents(processed_docs)

    # Save to disk
    print(f"Saving BM25 retriever to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(retriever, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ BM25 index built and saved successfully!")
    print(f"  - Documents indexed: {len(processed_docs)}")
    print(f"  - Index size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    return retriever


def load_bm25_index(load_path: str) -> BM25Retriever:

    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {load_path}")

    print(f"Loading BM25 retriever from {load_path}...")

    with open(load_path, 'rb') as f:
        retriever = pickle.load(f)

    print(f"✓ BM25 retriever loaded successfully!")
    print(f"  - Documents: {len(retriever.docs)}")
    print(f"  - Default k: {retriever.k}")

    return retriever


def search_bm25_index(
        retriever: BM25Retriever,
        query: str,
        k: Optional[int] = None
) -> List[Document]:

    # Set k if provided
    original_k = retriever.k
    if k is not None:
        retriever.k = k

    # Perform search using LangChain's invoke method
    results = retriever.invoke(query)

    # Restore original k
    if k is not None:
        retriever.k = original_k

    return results


def get_full_content(doc: Document) -> str:

    # If original_content exists in metadata, return it
    # Otherwise return page_content (means no extraction was used)
    return doc.metadata.get('original_content', doc.page_content)


def inspect_bm25_index(index_path: str):
    """
    Inspect a saved BM25 index to see what's inside.
    """
    index_path = Path(index_path)

    if not index_path.exists():
        print(f"❌ Index file not found at {index_path}")
        return

    print("=" * 60)
    print("BM25 Index Inspection")
    print("=" * 60)

    # Load the retriever
    with open(index_path, 'rb') as f:
        retriever = pickle.load(f)

    # File info
    file_size_mb = index_path.stat().st_size / 1024 / 1024
    print(f"\n📁 File Info:")
    print(f"  - Path: {index_path}")
    print(f"  - Size: {file_size_mb:.2f} MB")

    # Retriever attributes
    print(f"\n🔍 Retriever Info:")
    print(f"  - Type: {type(retriever).__name__}")
    print(f"  - Default k: {retriever.k}")
    print(f"  - Number of documents: {len(retriever.docs)}")

    # BM25 vectorizer info
    print(f"\n📊 BM25 Vectorizer Info:")
    if hasattr(retriever, 'vectorizer'):
        vectorizer = retriever.vectorizer
        print(f"  - Type: {type(vectorizer).__name__}")
        print(f"  - Number of documents indexed: {len(vectorizer.doc_freqs)}")
        print(f"  - Average document length: {vectorizer.avgdl:.2f} tokens")
        print(f"  - Vocabulary size: {len(vectorizer.doc_freqs[0]) if vectorizer.doc_freqs else 0} unique terms")

    # Sample documents
    print(f"\n📄 Sample Documents (first 3):")
    for i, doc in enumerate(retriever.docs[:3], 1):
        print(f"\n  Document {i}:")
        print(f"    - page_content: {doc.page_content[:100]}...")
        print(f"    - metadata keys: {list(doc.metadata.keys())}")

        # Check if original content is stored
        if 'original_content' in doc.metadata:
            print(f"    - Has original_content: Yes (length: {len(doc.metadata['original_content'])} chars)")
            print(f"    - Original preview: {doc.metadata['original_content'][:500]}...")
        else:
            print(f"    - Has original_content: No")

    # Metadata statistics
    print(f"\n📈 Metadata Statistics:")
    all_metadata_keys = set()
    for doc in retriever.docs:
        all_metadata_keys.update(doc.metadata.keys())
    print(f"  - Unique metadata keys across all docs: {all_metadata_keys}")

    # Check what was indexed (page_content that BM25 searches on)
    print(f"\n🔎 What BM25 Searches On (first 3 documents):")
    for i, doc in enumerate(retriever.docs[:3], 1):
        print(f"\n  Document {i}:")
        print(f"    Indexed text (page_content): '{doc.page_content}'")

    print("\n" + "=" * 60)