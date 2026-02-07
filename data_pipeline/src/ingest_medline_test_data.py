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
load_dotenv()
current_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_script_dir, "..", "data", "medlineplus_test_articles", "json")

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
    # 1. טעינת הגדרות מה-Env
    # שים לב: המפתח חייב להיות ב-.env תחת השם OPENAI_API_KEY או שם אחר שתבחר
    llmod_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_MEDLINE_TEST_INDEX_NAME")

    if not llmod_api_key:
        raise ValueError("❌ Error: OPENAI_API_KEY is missing (This is your LLMOD key)")
    if not pinecone_key:
        raise ValueError("❌ Error: PINECONE_API_KEY is missing")

    # 2. איסוף הקבצים
    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {DATA_PATH}")
        exit()

    all_documents = []
    print(f"📂 Processing {len(json_files)} files...")
    for file_path in tqdm(json_files, desc="Parsing JSONs"):
        docs = process_json_file(file_path)
        all_documents.extend(docs)

    print(f"\n📦 Prepared {len(all_documents)} chunks for ingestion.")

    # 3. אתחול מודל ה-Embedding עם הקונפיגורציה של LLMOD
    print("🔌 Initializing Embeddings via LLMOD.AI...")

    embeddings = OpenAIEmbeddings(
        model="RPRTHPB-text-embedding-3-small",  # המודל הספציפי שביקשת
        openai_api_key=llmod_api_key,  # המפתח של LLMOD
        openai_api_base="https://api.llmod.ai/v1",  # ה-Base URL של LLMOD
        check_embedding_ctx_length=False  # ביטול בדיקות אורך ספציפיות ל-OpenAI המקורי
    )

    # 4. שיגור ל-Pinecone
    print(f"🚀 Uploading to Pinecone Index: '{index_name}'...")

    try:
        PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            index_name=index_name
        )
        print("\n✅ Ingestion Complete! Vectors are indexed via LLMOD.")

    except Exception as e:
        print(f"\n❌ Error during upload: {e}")