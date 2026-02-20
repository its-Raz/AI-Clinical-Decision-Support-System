"""
frontend/app.py — Clinical Portal UI

Patient-centric interface:
  1. Patient selection on startup
  2. Personal clinical chat interface
  3. Natural language request processing
  4. Professional clinical design

Run with:
    streamlit run frontend/app.py
"""

import sys
import os
import tempfile
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from backend.agents.graph import build_system, analyze_existing_test
from backend.supabase.supabase_client import get_patients_summary, get_patient_lab_history, fetch_patient_by_id

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS for Clinical Look
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 8px 0;
        padding: 12px;
    }
    
    /* Patient selector */
    .patient-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    /* Input box */
    .stChatInputContainer {
        border-radius: 12px;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────

if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = None

if "current_patient_data" not in st.session_state:
    st.session_state.current_patient_data = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "trace_messages" not in st.session_state:
    st.session_state.trace_messages = []

if "running" not in st.session_state:
    st.session_state.running = False

if "last_state" not in st.session_state:
    st.session_state.last_state = None

if "awaiting_image_upload" not in st.session_state:
    st.session_state.awaiting_image_upload = False

# ─────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────

def classify_intent(user_message: str) -> dict:
    """
    Classify user's intent from their message.

    Returns:
        {"intent": "blood_test|lesion|unknown", "confidence": float}
    """
    msg_lower = user_message.lower()

    # Blood test keywords
    blood_keywords = ["blood test", "lab result", "test result", "recent test",
                      "lab work", "blood work", "analyze test", "check results",
                      "glucose", "hemoglobin", "creatinine"]

    # Lesion keywords
    lesion_keywords = ["lesion", "skin", "mole", "spot", "rash", "growth",
                       "analyze image", "check skin", "dermatology", "picture"]

    blood_score = sum(1 for kw in blood_keywords if kw in msg_lower)
    lesion_score = sum(1 for kw in lesion_keywords if kw in msg_lower)

    if blood_score > lesion_score and blood_score > 0:
        return {"intent": "blood_test", "confidence": blood_score}
    elif lesion_score > blood_score and lesion_score > 0:
        return {"intent": "lesion", "confidence": lesion_score}
    else:
        return {"intent": "unknown", "confidence": 0}


def _split_messages(all_messages: list):
    """Separate trace from patient-facing messages."""
    patient_msgs = [m for m in all_messages if m.get("role") == "assistant"]
    trace_msgs = all_messages
    return patient_msgs, trace_msgs


def _run_pipeline(initial_state: dict, trigger_description: str):
    """Execute analysis pipeline."""
    st.session_state.running = True

    existing_chat = st.session_state.chat_messages.copy()
    st.session_state.trace_messages = []
    st.session_state.last_state = None

    with st.spinner("🤖 Analyzing... this may take 30–60 seconds..."):
        try:
            st.session_state.trace_messages.append({
                "role": "system",
                "content": trigger_description,
            })

            system = build_system()
            final_state = system.run(initial_state)

            all_msgs = final_state.get("messages", [])
            patient_msgs, trace_msgs = _split_messages(all_msgs)

            st.session_state.chat_messages = existing_chat + patient_msgs
            st.session_state.trace_messages = (
                st.session_state.trace_messages + trace_msgs
            )
            st.session_state.last_state = final_state

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.exception(e)
        finally:
            st.session_state.running = False
            st.session_state.awaiting_image_upload = False


def render_trace_message(msg: dict, index: int):
    """Render trace message in system panel."""
    role = msg.get("role", "system")
    content = msg.get("content", "")

    role_colors = {
        "system": "#2563eb",
        "tool": "#059669",
        "assistant": "#7c3aed",
    }

    color = role_colors.get(role, "#64748b")

    st.markdown(f"""
    <div style="
        background: white;
        border-left: 4px solid {color};
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    ">
        <div style="color: {color}; font-weight: 600; margin-bottom: 4px;">
            #{index} {role.upper()}
        </div>
        <div style="color: #374151;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_comparison_image(image_path: str, prediction: dict) -> Image.Image | None:
    """Create visualization with GT and prediction boxes."""
    try:
        from backend.agents.skin_care_analyst.data import test_labels

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Draw prediction box (orange)
        if prediction and "bbox" in prediction:
            x1, y1, x2, y2 = prediction["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=4)

            label = f"{prediction.get('label', 'Unknown')} ({prediction.get('conf', 0):.1%})"
            draw.text((x1, y2 + 5), label, fill=(255, 165, 0), font=font)

        return img
    except:
        return Image.open(image_path)


# ─────────────────────────────────────────────────────────────────────────
# Main App Logic
# ─────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #1e3a8a; margin-bottom: 5px;">🏥 Clinical Portal</h1>
    <p style="color: #64748b; font-size: 1rem;">Your Personal Health Assistant</p>
</div>
<hr style="border: none; border-top: 2px solid #e2e8f0; margin: 20px 0;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# PATIENT SELECTION (if no patient selected)
# ─────────────────────────────────────────────────────────────────────────

if not st.session_state.current_patient_id:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### 👤 Patient Login")
        st.markdown("Select your profile to access your clinical portal:")

        with st.spinner("Loading patient database..."):
            try:
                patients = get_patients_summary()

                if not patients:
                    st.error("⚠️ No patients found in database.")
                else:
                    # Create patient cards
                    for patient in patients[:20]:  # Show first 20 for demo
                        with st.container():
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.markdown(f"""
                                <div class="patient-card">
                                    <h4 style="margin: 0; color: #1e3a8a;">{patient['name']}</h4>
                                    <p style="margin: 5px 0 0 0; color: #64748b; font-size: 0.9rem;">
                                        ID: {patient['id']} | Age: {patient['age']} | Tests: {patient['test_count']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                if st.button("Login", key=f"login_{patient['id']}",
                                           type="primary", use_container_width=True):
                                    st.session_state.current_patient_id = patient['id']
                                    st.session_state.current_patient_data = fetch_patient_by_id(patient['id'])

                                    # Initialize welcome message
                                    patient_name = st.session_state.current_patient_data['name'].split()[0]
                                    welcome_msg = {
                                        "role": "assistant",
                                        "content": (
                                            f"Hi **{patient_name}**, what can I do for you today?\n\n"
                                            "I can help you with:\n"
                                            "- **Analyze your recent blood test results**\n"
                                            "- **Analyze a skin lesion image**\n\n"
                                            "Just type your request below!"
                                        )
                                    }
                                    st.session_state.chat_messages = [welcome_msg]
                                    st.rerun()

            except Exception as e:
                st.error(f"Error loading patients: {e}")
                st.exception(e)

    with right_col:
        st.markdown("### ⚙️ System Status")
        st.info("👈 Please select a patient to access the clinical portal")
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="color: #1e3a8a;">System Features</h4>
            <ul style="color: #64748b;">
                <li>AI-powered lab result analysis</li>
                <li>Skin lesion screening</li>
                <li>Personalized health recommendations</li>
                <li>Secure patient data access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# CLINICAL CHAT INTERFACE (after patient selected)
# ─────────────────────────────────────────────────────────────────────────

else:
    patient_data = st.session_state.current_patient_data
    patient_name = patient_data['name']
    patient_first_name = patient_name.split()[0]

    # Logout button in top right
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown(f"### Welcome, {patient_first_name}!")
    with col2:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_patient_id = None
            st.session_state.current_patient_data = None
            st.session_state.chat_messages = []
            st.session_state.trace_messages = []
            st.session_state.last_state = None
            st.rerun()

    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    # Split screen: Chat | System Trace
    left_col, right_col = st.columns([1, 1])

    # ── LEFT: Clinical Chat ───────────────────────────────────────────────
    with left_col:
        st.markdown("### 💬 Your Health Assistant")

        # Chat container
        chat_container = st.container(height=500, border=True)

        with chat_container:
            for msg in st.session_state.chat_messages:
                role = msg["role"]
                content = msg["content"]

                with st.chat_message(role, avatar="🏥" if role == "assistant" else "👤"):
                    st.markdown(content)

        # Image upload if awaiting
        if st.session_state.awaiting_image_upload:
            st.markdown("### 📤 Upload Lesion Image")
            uploaded_file = st.file_uploader(
                "Choose an image (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                key="lesion_upload",
            )

            if uploaded_file:
                st.image(uploaded_file, width=300)

                if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                    # Save to temp
                    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Build state
                    initial_state = {
                        "request_type": "image_lesion_analysis",
                        "patient_id": st.session_state.current_patient_id,
                        "lab_result": None,
                        "lab_insights": None,
                        "image_path": temp_path,
                        "vision_results": None,
                        "vision_insights": None,
                        "messages": [],
                        "next_step": "",
                        "final_report": None,
                    }

                    _run_pipeline(
                        initial_state,
                        f"[Skin Lesion Analysis] Patient {st.session_state.current_patient_id} uploaded image: {uploaded_file.name}"
                    )
                    st.rerun()

        # Chat input
        user_input = st.chat_input(
            "Type your request here...",
            disabled=st.session_state.running or st.session_state.awaiting_image_upload,
        )

        if user_input:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })

            # Classify intent
            intent_result = classify_intent(user_input)
            intent = intent_result["intent"]

            if intent == "blood_test":
                # Trigger blood test analysis on most recent test
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "📋 Understood! Let me analyze your most recent blood test results..."
                })

                lab_history = get_patient_lab_history(st.session_state.current_patient_id)

                if not lab_history:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "⚠️ I couldn't find any lab test results in your history. Please contact your healthcare provider."
                    })
                    st.rerun()
                else:
                    initial_state = analyze_existing_test(
                        st.session_state.current_patient_id,
                        -1  # Most recent
                    )

                    _run_pipeline(
                        initial_state,
                        f"[Blood Test Analysis] Patient {st.session_state.current_patient_id} requested analysis of recent test"
                    )
                    st.rerun()

            elif intent == "lesion":
                # Request image upload
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "🩺 Sure! Please upload a clear image of the lesion below so I can analyze it."
                })
                st.session_state.awaiting_image_upload = True
                st.rerun()

            else:
                # Unknown intent
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": (
                        "I'm sorry, I can't assist with that request. "
                        "I can help you with:\n"
                        "- Analyzing your **blood test results**\n"
                        "- Analyzing a **skin lesion**\n\n"
                        "Please rephrase your request!"
                    )
                })
                st.rerun()

    # ── RIGHT: System Trace ───────────────────────────────────────────────
    with right_col:
        st.markdown("### ⚙️ System Trace")

        trace_container = st.container(height=500, border=True)

        with trace_container:
            if not st.session_state.trace_messages:
                st.markdown("""
                <p style="text-align: center; color: #64748b; margin-top: 200px;">
                    No system activity yet.
                </p>
                """, unsafe_allow_html=True)
            else:
                for i, msg in enumerate(st.session_state.trace_messages, 1):
                    render_trace_message(msg, i)

                # Visualization for skin analysis
                if st.session_state.last_state:
                    req_type = st.session_state.last_state.get("request_type")

                    if req_type == "image_lesion_analysis":
                        image_path = st.session_state.last_state.get("image_path")
                        vision_results = st.session_state.last_state.get("vision_results")

                        if image_path and vision_results:
                            st.markdown("---")
                            st.markdown("**📊 Analysis Result**")

                            comparison_img = create_comparison_image(image_path, vision_results)
                            if comparison_img:
                                st.image(comparison_img, use_container_width=True)

        # Patient info panel
        with st.expander("👤 Patient Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Patient ID", patient_data['id'])
                st.metric("Age", patient_data.get('age', 'N/A'))
            with col2:
                st.metric("Sex", patient_data.get('sex', 'N/A'))
                lab_count = len(patient_data.get('lab_history', []))
                st.metric("Lab Tests", lab_count)

            if patient_data.get('chronic_conditions'):
                st.markdown("**Chronic Conditions:**")
                for condition in patient_data['chronic_conditions']:
                    st.markdown(f"- {condition}")

# ─────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────

st.markdown("<hr style='margin-top: 30px;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    🔒 Secure Clinical Portal | HIPAA Compliant | 24/7 AI Support
</div>
""", unsafe_allow_html=True)