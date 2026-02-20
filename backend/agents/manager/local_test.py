# test_evidence_route.py

import os
from dotenv import load_dotenv

# ודא שמשתני הסביבה נטענים (בעיקר OPENAI_API_KEY)
load_dotenv()

# ייבוא המנהל הראשי (יש לוודא שהנתיב תואם למבנה התיקיות שלך)
from backend.agents.manager import ManagerAgent
from backend.agents.global_state import AgentState


def run_evidence_integration_test():
    print("🚀 Starting Evidence Analyst Integration Test...\n")

    # 1. יצירת מצב גלובלי (State) מזויף עם שאלת משתמש
    mock_state: AgentState = {
        "request_type": "evidence_analyst",  # מנתב ישירות לסוכן החדש
        "patient_id": "TEST-EVIDENCE-99",
        "lab_result": None,
        "lab_insights": None,
        "image_path": None,
        "vision_results": None,
        "vision_insights": None,
        "evidence_insights": None,  # השדה החדש שהוספנו
        "next_step": "",
        "final_report": None,
        "messages": [
            {
                "role": "system",
                "content": "You are a medical system."
            },
            {
                "role": "user",
                # השאלה שהסוכן ReAct אמור לקבל
                "content": "What are the common causes of low hemoglobin, and what dietary changes can help?"
            }
        ]
    }

    try:
        # 2. אתחול המנהל (שבונה את כל הגרף)
        manager = ManagerAgent()

        # 3. הרצת הגרף עם המצב המזויף
        print("\n⏳ Invoking the Manager Graph...")
        final_state = manager.run(mock_state)

        # 4. הדפסת התוצאות לבדיקה
        print("\n" + "🎯 " + "=" * 47)
        print("                 TEST RESULTS")
        print("=" * 50)

        print(f"\n✅ Final Next Step Resolved: {final_state.get('next_step')}")

        print("\n🧠 1. RAW EVIDENCE INSIGHTS (From ReAct Agent):")
        print("-" * 50)
        print(final_state.get("evidence_insights", "❌ No insights found!"))

        print("\n💬 2. FINAL PATIENT REPORT (From Deliver Node):")
        print("-" * 50)
        print(final_state.get("final_report", "❌ No final report generated!"))
        print("\n" + "=" * 50)

    except Exception as e:
        print(f"\n❌ Test Failed with error: {e}")


if __name__ == "__main__":
    run_evidence_integration_test()