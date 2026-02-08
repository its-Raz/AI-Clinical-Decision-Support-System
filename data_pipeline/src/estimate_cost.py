import os
import glob
import json
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# הגדרות מחיר (מעודכן ל-2024 עבור text-embedding-3-small)
PRICE_PER_1M_TOKENS = 0.02
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "medlineplus_test_articles", "json")

# text-embedding-3-small
def estimate_medline_test_data_embedding():

    encoder = tiktoken.get_encoding("cl100k_base")

    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    total_tokens = 0
    total_files = len(json_files)

    print(f"📊 Analyzing {total_files} files...")

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # אנחנו סופרים את הטקסט הגולמי + הכותרות (בערך מה שנשלח למודל)
            # זו הערכה, כי ב-Chunking אנחנו מוסיפים קצת חפיפה ומטא-דאטה
            full_text = data.get("title", "")
            for section in data.get("sections", []):


                full_text += " " + section.get("title", "")
                full_text += " " + " ".join(section.get("content", []))

            # ספירת טוקנים אמיתית
            tokens = len(encoder.encode(full_text))
            total_tokens += tokens

    # חישוב עלות
    estimated_cost = (total_tokens / 1_000_000) * PRICE_PER_1M_TOKENS

    # פקטור ביטחון של 20% (בגלל Overlap ומטא-דאטה שנוסיף ב-Ingest)
    safe_estimate = estimated_cost * 1.2

    print("\n--- Cost Estimation Report ---")
    print(f"Total Files: {total_files}")
    print(f"Total Raw Tokens: {total_tokens:,}")
    print(f"Model: text-embedding-3-small")
    print(f"Estimated Cost: ${estimated_cost:.4f}")
    print(f"Safe Estimate (w/ Overlap): ${safe_estimate:.4f}")
    print("------------------------------")


if __name__ == "__main__":
    estimate_medline_test_data_embedding()