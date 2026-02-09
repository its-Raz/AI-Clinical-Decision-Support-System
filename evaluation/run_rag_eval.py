import json
import os
import csv
from dotenv import load_dotenv
from pinecone import Pinecone
from backend.tools.medline_test_rag import create_medline_test_rag
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision,IDBasedContextRecall
import numpy as np
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
load_dotenv()
# --- Configuration ---
MEDLINE_TEST_GOLDEN_SET_PATH = 'data/medline_test_golden_dataset.json'
# Path to the CSV you just created/uploaded
PINECONE_CSV_PATH = 'data/pinecone_first_200_sorted.csv'


async def calculate_ID_Precision_score(retrieved_ids, ground_truth_ids):
    # 1. הכנת הנתונים
    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=ground_truth_ids
    )


    id_precision = IDBasedContextPrecision()


    score = await id_precision.single_turn_ascore(sample)
    print(f"Context Precision Score: {score}")
    return score

async def calculate_ID_Recall_score(retrieved_ids, ground_truth_ids):
    # 1. הכנת הנתונים
    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=ground_truth_ids
    )


    id_recall = IDBasedContextRecall()
    await id_recall.single_turn_ascore(sample)


    score = await id_recall.single_turn_ascore(sample)
    print(f"Context Recall Score: {score}")
    return score

    return score
def load_id_lookup_table(csv_path):
    """
    Loads the CSV and creates a dictionary for fast ID lookup.
    Key: (doc_title, sec_title, chunk_index) -> Value: pinecone_id
    """
    try:
        df = pd.read_csv(csv_path)
        lookup = {}

        for _, row in df.iterrows():
            # Normalize keys:
            # 1. Strip whitespace from titles
            # 2. Convert chunk_index to integer (handles 0.0 vs 0)
            d_title = str(row['doc_title']).strip()
            s_title = str(row['sec_title']).strip()
            c_index = int(row['chunk_index']) if pd.notna(row['chunk_index']) else 0

            key = (d_title, s_title, c_index)
            lookup[key] = row['pinecone_id']

        print(f"✓ Loaded lookup table with {len(lookup)} IDs from {csv_path}")
        return lookup

    except Exception as e:
        print(f"❌ Error loading CSV lookup table: {e}")
        return {}


async def medline_test_rag_evaluation(golden_set_path, lookup_csv_path,bm25_weight,vector_weight):
    # 1. Load the ID Lookup Table
    id_map = load_id_lookup_table(lookup_csv_path)
    precisions =[]
    recalls =[]
    try:
        # 2. Open the Golden Dataset
        with open(golden_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Successfully loaded {len(data)} test questions.\n")

        # 3. Initialize RAG
        rag = create_medline_test_rag()
        rag.ensemble_retriever.weights = [bm25_weight, vector_weight]
        print(f"Running Eval with weights: BM25={bm25_weight}, Vector={vector_weight}...\n")
        # 4. Iterate through questions
        for i, item in enumerate(data):

            question = item.get("question", "N/A")
            ground_truth_ids = item.get("ground_truth_ids", [])

            print(f"\nExample {i + 1}:")
            print(f"  Question: {question}")
            print(f"  Ground Truth IDs: {ground_truth_ids}")

            # --- Perform Retrieval ---
            # Retrieve documents (Document objects)
            retrieved_docs = rag.query(question)

            # --- Extract IDs ---
            retrieved_ids = []
            for i,doc in enumerate(retrieved_docs):
                # Get metadata from the retrieved document
                # Note: Adjust key names ('Doc_Title' vs 'doc_title') based on your specific RAG implementation
                r_doc_title = str(doc.metadata.get('Doc_Title', '')).strip()
                r_sec_title = str(doc.metadata.get('Sec_Title', '')).strip()

                # Handle chunk index safely
                raw_idx = doc.metadata.get('Chunk_Index', 0)

                try:
                    r_chunk_index = int(float(raw_idx))
                except:
                    r_chunk_index = 0

                # Look up the ID
                lookup_key = (r_doc_title, r_sec_title, r_chunk_index)

                found_id = id_map.get(lookup_key, f"ID {i} NOT FROM 200 LIST")

                retrieved_ids.append(found_id)

            print(f"  Retrieved IDs:    {retrieved_ids}")

            prec_score = await calculate_ID_Precision_score(retrieved_ids, ground_truth_ids)
            recall_score = await calculate_ID_Recall_score(retrieved_ids, ground_truth_ids)
            precisions.append(prec_score)
            recalls.append(recall_score)



        mean_p = np.mean(precisions) if precisions else 0.0
        median_p = np.median(precisions) if precisions else 0.0
        mean_r = np.mean(recalls) if recalls else 0.0
        median_r = np.median(recalls) if recalls else 0.0

        print(f"  -> Done. Mean P: {mean_p:.3f}, Median P: {median_p:.3f}\n")
        print(f"  -> Done. Mean R: {mean_r:.3f}, Median R: {median_r:.3f}\n")
        return mean_p, median_p, mean_r, median_r


    except FileNotFoundError:
        print(f"Error: File not found.")
        return 0.0,0.0,0.0,0.0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 0.0, 0.0,0.0,0.0















async def run_experiments_and_plot():
    # הגדרות נתיבים
    MEDLINE_TEST_GOLDEN_SET_PATH = 'data/medline_test_golden_dataset.json'
    PINECONE_CSV_PATH = 'data/pinecone_first_200_sorted.csv'

    # שלושת הקונפיגורציות
    configs = [
        (1.0, 0.0, "BM25 Only"),
        (0.0, 1.0, "Vector Only"),
        (0.5, 0.5, "Hybrid (RRF)")
    ]

    # שמירת התוצאות
    results = {
        "Mean Precision": [],
        "Median Precision": [],
        "Mean Recall": [],
        "Median Recall": []
    }
    labels = []

    print("🚀 Starting 3-Run Experiment...")
    print("=" * 40)

    for bm, vec, label in configs:
        labels.append(label)
        # הרצת ההערכה
        mean_p, median_p, mean_r, median_r = await medline_test_rag_evaluation(
            MEDLINE_TEST_GOLDEN_SET_PATH,
            PINECONE_CSV_PATH,
            bm,
            vec
        )

        results["Mean Precision"].append(mean_p)
        results["Median Precision"].append(median_p)
        results["Mean Recall"].append(mean_r)
        results["Median Recall"].append(median_r)

    # --- ויזואליזציה (Matplotlib) ---
    print("📊 Generating Plot...")

    x = np.arange(len(labels))  # מיקומי ה-X לקבוצות
    width = 0.2  # רוחב העמודה

    fig, ax = plt.subplots(figsize=(12, 6))

    # ציור 4 עמודות לכל קונפיגורציה
    rects1 = ax.bar(x - 1.5 * width, results["Mean Precision"], width, label='Mean Precision', color='skyblue')
    rects2 = ax.bar(x - 0.5 * width, results["Median Precision"], width, label='Median Precision', color='steelblue')
    rects3 = ax.bar(x + 0.5 * width, results["Mean Recall"], width, label='Mean Recall', color='lightgreen')
    rects4 = ax.bar(x + 1.5 * width, results["Median Recall"], width, label='Median Recall', color='forestgreen')

    # עיצוב הגרף
    ax.set_ylabel('Score (0-1)')
    ax.set_title('RAG Evaluation: BM25 vs Vector vs Hybrid')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)  # קצת רווח למעלה לתוויות
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # הוספת מספרים מעל העמודות
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    output_file = 'rag_evaluation_results.png'
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")

    # הצגת התמונה (אם רצים ב-Notebook)
    plt.show()


# --- הרצה ---
if __name__ == "__main__":
    asyncio.run(run_experiments_and_plot())
