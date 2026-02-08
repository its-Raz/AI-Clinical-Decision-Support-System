import os
import json
import glob
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils import *
from bm25_reitriver import *

load_dotenv()
current_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_script_dir, "..", "data", "medlineplus_test_articles", "json")

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIM = 512
CHUNK_SIZE = 1000
OVERLAP_PERCENTAGE = 0.20
chunk_overlap_int = int(CHUNK_SIZE * OVERLAP_PERCENTAGE)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=chunk_overlap_int,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

def process_json_file(file_path):
    """

    """
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_title = data.get("title", "Unknown")

    for section in data.get("sections", []):
        sec_title = section.get("title", "General")

        # דילוג על רפרנסים (לא רלוונטי ל-RAG)
        if sec_title.lower() == "references" or sec_title.lower() == "related health topics" or sec_title.lower() == "related medical tests":
            continue

        # איחוד כל הטקסטים ברשימה לטקסט אחד
        content_text = " ".join(section.get("content", []))
        if len(content_text) == 0:
            print(f'Length of section is 0, Continue')
            continue
        # Case A: Short Section (< X chars) -> NO SPLIT
        if len(content_text) < CHUNK_SIZE:
            # בניית הפורמט המיוחד שביקשת
            formatted_text = (
                f"File Title: {doc_title}\n"
                f"Sub Title: {sec_title}\n"
                f"Content splitted: no\n"
                f"Section: {content_text}"
            )


            doc = Document(
                page_content=formatted_text,
                metadata={

                    "Doc_title": doc_title,
                    "Sec_title": sec_title,
                    "Splitted": "no",
                    "Original_Length": len(content_text)
                }
            )
            documents.append(doc)

        # Case B: Long Section (> X chars) -> SPLIT WITH OVERLAP
        else:

            chunks = text_splitter.split_text(content_text)

            for i, chunk in enumerate(chunks):
                # פורמט זהה, רק עם התוכן החתוך
                formatted_text = (
                    f"File Title: {doc_title}\n"
                    f"Sub Title: {sec_title}\n"
                    f"Content Splitted: yes\n"
                    f"Section: {chunk}"
                )

                doc = Document(
                    page_content=formatted_text,
                    metadata={
                        "Doc_Title": doc_title,
                        "Sec_Title": sec_title,
                        "Splitted": "yes",
                        "Chunk_Index": i
                    }
                )
                documents.append(doc)

    return documents


if __name__ == "__main__":
    # Define BM25 index path
    BM25_INDEX_PATH = os.path.join(current_script_dir, "..", "data", "bm25", "medline_test_bm25.pkl")
    llmod_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_MEDLINE_TEST_INDEX_NAME")

    if not llmod_api_key:
        raise ValueError("❌ Error: OPENAI_API_KEY is missing (This is your LLMOD key)")
    if not pinecone_key:
        raise ValueError("❌ Error: PINECONE_API_KEY is missing")


    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {DATA_PATH}")
        exit()

    all_documents = []
    print(f"Processing {len(json_files)} files...")
    for file_path in tqdm(json_files, desc="Parsing JSONs"):
        docs = process_json_file(file_path)
        all_documents.extend(docs)

    print(f"\nPrepared {len(all_documents)} chunks for ingestion.")
    max_tokens=0
    for doc in all_documents:
        tokens = estimate_tokens_number(doc.page_content)
        if tokens > max_tokens:
            max_tokens = tokens
    print(f'Max Tokens: {max_tokens}')
    # ============================================================
    # 1. Build and Save BM25 Index
    # ============================================================
    print("\n" + "=" * 60)
    print("Building BM25 Index...")
    print("=" * 60)


    def extract_titles_from_metadata(doc: Document) -> str:
        """Extract titles from metadata for BM25 searching"""
        # Use the correct keys with capital T
        title = doc.metadata.get('Doc_Title', '')  # Changed from Doc_title
        subtitle = doc.metadata.get('Sec_Title', '')  # Changed from Sec_title
        return f"{title} {subtitle}".strip()


    bm25_retriever = build_and_save_bm25_index(
        documents=all_documents,
        save_path=BM25_INDEX_PATH,
        text_extraction_fn=extract_titles_from_metadata,
        overwrite=True
    )

    print("✓ BM25 index created and saved successfully!")
    inspect_bm25_index(BM25_INDEX_PATH)
    # ============================================================
    # 2. Ingest to Pinecone
    # ============================================================

    # print("Initializing Embeddings via LLMOD.AI...")
    #
    # embeddings = OpenAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     openai_api_key=llmod_api_key,
    #     base_url="https://api.llmod.ai/v1",
    #     check_embedding_ctx_length=True,
    #     dimensions = EMBEDDING_DIM
    # )
    #
    # print(f'Get index {index_name}')
    # index = get_pinecone_index(
    #     PINECONE_API_KEY=pinecone_key,
    #     PINECONE_INDEX_NAME=index_name,
    #     embedding_dim=EMBEDDING_DIM
    # )
    # print(f"Uploading to Pinecone Index: '{index_name}'...")
    # try:
    #     PineconeVectorStore.from_documents(
    #         documents=all_documents,
    #         embedding=embeddings,
    #         index_name=index_name
    #     )
    #     print("\nIngestion Complete! Vectors are indexed via LLMOD.")
    #
    # except Exception as e:
    #     print(f"\nError during upload: {e}")