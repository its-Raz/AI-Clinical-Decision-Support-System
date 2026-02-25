import os
import sys
from datetime import datetime

# הוספת נתיב הפרויקט כדי שיוכל למצוא את המודול של Supabase
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.supabase.supabase_client import add_lab_result


def insert_abnormal_test():
    patient_id = "P001"

    # הגדרת הבדיקה החדשה:
    # Glucose: 178 (High - סוכרת לא מאוזנת)
    # Hemoglobin: 10.2 (Low - אנמיה)
    # Creatinine: 1.4 (High - תפקוד כליות גבולי)
    new_lab_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "Glucose": {
            "value": 178,
            "unit": "mg/dL",
            "flag": "high"
        },
        "Hemoglobin": {
            "value": 10.2,
            "unit": "g/dL",
            "flag": "low"
        },
        "Creatinine": {
            "value": 1.4,
            "unit": "mg/dL",
            "flag": "high"
        }
    }

    print(f"🚀 Attempting to add abnormal lab results for {patient_id}...")

    success = add_lab_result(patient_id, new_lab_entry)

    if success:
        print("✅ New lab results were successfully added to Supabase!")
    else:
        print("❌ Failed to add lab results. Check your Supabase credentials and Patient ID.")


if __name__ == "__main__":
    insert_abnormal_test()