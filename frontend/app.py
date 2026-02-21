"""
frontend/app.py — Clinical Portal UI

Patient-centric interface:
  1. Eager backend initialisation with friendly startup UI
  2. Patient selection
  3. Semantic routing + Judge LLM for intent classification
  4. Clinical chat with blood test, skin lesion, and evidence Q&A flows

Run with:
    streamlit run frontend/app.py

Bug fixes applied:
  - Startup shows a friendly st.status() box instead of raw "Running _startup()"
  - User message appears immediately on submit (two-phase rerun pattern)
  - Chat shows a processing banner while pipeline executes
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── Single import point for all backend logic ─────────────────────────────
from backend.main import (
    initialize,
    route_request,
    build_blood_test_state,
    build_lesion_state,
    build_evidence_state,
    execute_pipeline,
    analyze_existing_test,
)
from backend.supabase.supabase_client import (
    get_patients_summary,
    get_patient_lab_history,
    fetch_patient_by_id,
)

# ─────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 8px 0;
        padding: 12px;
    }
    .patient-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stChatInputContainer {
        border-radius: 12px;
    }
    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Header  — rendered BEFORE startup so the page is never blank
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #1e3a8a; margin-bottom: 5px;">🏥 Clinical Portal</h1>
    <p style="color: #64748b; font-size: 1rem;">Your Personal Health Assistant</p>
</div>
<hr style="border: none; border-top: 2px solid #e2e8f0; margin: 20px 0;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Eager backend initialisation  (BUG 1 FIX)
#
# @st.cache_resource ensures initialize() runs exactly once per server
# process. We call it AFTER the header so the page is never blank.
#
# On cold start: st.status() shows a friendly progress box while the
# router builds its centroid index and the agent graph compiles.
# On warm reruns: _startup() returns from cache in microseconds and the
# status box opens and closes before the user can see it.
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _startup() -> bool:
    """Build router and ManagerAgent once at server startup."""
    initialize()
    return True


with st.status("⚙️  Starting Clinical Portal…", expanded=True) as _boot_status:
    st.write("🧭  Building semantic router — embedding route examples…")
    st.write("🏗️  Compiling agent graph…")
    _startup()
    _boot_status.update(
        label="✅  System ready!",
        state="complete",
        expanded=False,
    )

# ─────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────

defaults = {
    "current_patient_id":    None,
    "current_patient_data":  None,
    "chat_messages":         [],
    "trace_messages":        [],
    # True while the agent pipeline is executing
    "running":               False,
    "last_state":            None,
    "awaiting_image_upload": False,
    # Stores {user_input, route_result} between the two-phase reruns.
    # Phase 1: user submits → message shown + pending_pipeline set → rerun
    # Phase 2: pipeline executes → results stored → rerun
    "pending_pipeline":      None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────

def _split_messages(all_messages: list):
    """Separate patient-facing (assistant) messages from the full trace."""
    patient_msgs = [m for m in all_messages if m.get("role") == "assistant"]
    return patient_msgs, all_messages


def _run_pipeline(initial_state: dict, trigger_description: str) -> None:
    """
    Execute the backend pipeline and store results in session state.

    Called during Phase 2 of the two-phase rerun pattern, AFTER the chat
    container has already been rendered and is visible to the user.
    `running` is already True at this point (set during Phase 1).
    """
    existing_chat = st.session_state.chat_messages.copy()
    st.session_state.trace_messages = []
    st.session_state.last_state = None

    try:
        st.session_state.trace_messages.append({
            "role":    "system",
            "content": trigger_description,
        })

        final_state = execute_pipeline(initial_state)

        patient_msgs, trace_msgs = _split_messages(
            final_state.get("messages", [])
        )

        st.session_state.chat_messages = existing_chat + patient_msgs
        st.session_state.trace_messages += trace_msgs
        st.session_state.last_state = final_state

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.exception(e)
    finally:
        st.session_state.running               = False
        st.session_state.awaiting_image_upload = False


def _execute_pending_pipeline(pending: dict) -> None:
    """
    Dispatch the pending pipeline based on the stored route result.
    Called during Phase 2 after the chat container is already rendered.
    """
    user_input        = pending["user_input"]
    route_result      = pending["route_result"]
    proposed_category = route_result["category"]
    router_score      = route_result["score"]
    router_confidence = route_result["confidence"]
    patient_id        = st.session_state.current_patient_id

    if proposed_category == "blood_test_analysis":
        lab_state = analyze_existing_test(patient_id, -1)
        initial_state = build_blood_test_state(
            user_text         = user_input,
            proposed_category = proposed_category,
            router_score      = router_score,
            router_confidence = router_confidence,
            patient_id        = patient_id,
            lab_result        = lab_state.get("lab_result", []),
        )
        _run_pipeline(
            initial_state,
            f"[Blood Test Analysis] Patient {patient_id} | "
            f"Router: {proposed_category} ({router_score:.4f}, {router_confidence})",
        )

    elif proposed_category == "evidence_analyst":
        initial_state = build_evidence_state(
            user_text         = user_input,
            proposed_category = proposed_category,
            router_score      = router_score,
            router_confidence = router_confidence,
            patient_id        = patient_id,
        )
        _run_pipeline(
            initial_state,
            f"[Evidence Analyst] Patient {patient_id} | "
            f"Router: {proposed_category} ({router_score:.4f}, {router_confidence})",
        )

    else:
        # Should not reach here — unknown category, clear running state
        st.session_state.running = False


def render_trace_message(msg: dict, index: int) -> None:
    """Render one trace message in the system panel."""
    role  = msg.get("role", "system")
    content = msg.get("content", "")

    color = {
        "system":    "#2563eb",
        "tool":      "#059669",
        "assistant": "#7c3aed",
    }.get(role, "#64748b")

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
        <div style="color: #374151;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def create_comparison_image(image_path: str, prediction: dict) -> Image.Image | None:
    """Draw the YOLO prediction bounding box onto the original image."""
    try:
        img  = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        if prediction and "bbox" in prediction:
            x1, y1, x2, y2 = prediction["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=4)
            label = f"{prediction.get('label', 'Unknown')} ({prediction.get('conf', 0):.1%})"
            draw.text((x1, y2 + 5), label, fill=(255, 165, 0), font=font)

        return img
    except Exception:
        try:
            return Image.open(image_path)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────
# PATIENT SELECTION SCREEN
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
                    for patient in patients[:20]:
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
                            if st.button(
                                "Login",
                                key=f"login_{patient['id']}",
                                type="primary",
                                use_container_width=True,
                            ):
                                st.session_state.current_patient_id   = patient["id"]
                                st.session_state.current_patient_data = fetch_patient_by_id(patient["id"])
                                first_name = st.session_state.current_patient_data["name"].split()[0]
                                st.session_state.chat_messages = [{
                                    "role":    "assistant",
                                    "content": (
                                        f"Hi **{first_name}**, what can I do for you today?\n\n"
                                        "I can help you with:\n"
                                        "- 🩸 **Analyze your recent blood test results**\n"
                                        "- 🔬 **Analyze a skin lesion image**\n"
                                        "- 💊 **Answer general medical questions**\n\n"
                                        "Just type your request below!"
                                    ),
                                }]
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
                <li>Medical Q&amp;A with evidence retrieval</li>
                <li>Personalized health recommendations</li>
                <li>Secure patient data access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# CLINICAL CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────

else:
    patient_data       = st.session_state.current_patient_data
    patient_first_name = patient_data["name"].split()[0]

    # ── Top bar ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown(f"### Welcome, {patient_first_name}!")
    with col2:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_patient_id   = None
            st.session_state.current_patient_data = None
            st.session_state.chat_messages         = []
            st.session_state.trace_messages        = []
            st.session_state.last_state            = None
            st.session_state.running               = False
            st.session_state.pending_pipeline      = None
            st.rerun()

    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1])

    # ── LEFT: Chat ────────────────────────────────────────────────────────
    with left_col:
        st.markdown("### 💬 Your Health Assistant")

        # ── PHASE 2: pipeline execution  (BUG 2 & 3 FIX) ─────────────────
        # If a pipeline is pending it means Phase 1 already ran:
        # the user's message and acknowledgment are in chat_messages and
        # will be rendered by the chat container below BEFORE this block
        # executes. So the user sees their message + acknowledgment + the
        # processing banner, then the pipeline runs, then st.rerun() shows
        # the final result.
        if st.session_state.pending_pipeline and st.session_state.running:
            pending = st.session_state.pending_pipeline
            st.session_state.pending_pipeline = None  # clear before executing

            # ── Chat container — renders immediately with user message ─────
            chat_container = st.container(height=500, border=True)
            with chat_container:
                for msg in st.session_state.chat_messages:
                    with st.chat_message(
                        msg["role"],
                        avatar="🏥" if msg["role"] == "assistant" else "👤",
                    ):
                        st.markdown(msg["content"])

            # ── Processing banner (BUG 3 FIX) ────────────────────────────
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #e0e7ff, #f0f4ff);
                border: 1px solid #6366f1;
                border-radius: 10px;
                padding: 14px 18px;
                margin: 8px 0;
                text-align: center;
            ">
                🔄 <strong style="color: #4338ca;">Processing your request…</strong><br>
                <span style="font-size: 0.82rem; color: #6366f1;">
                    This may take 30–60 seconds. The input is locked while analysis runs.
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Disabled chat input shown while processing
            st.chat_input("Processing…", disabled=True)

            # ── Execute the pipeline ──────────────────────────────────────
            with st.spinner("🤖 Agents working…"):
                _execute_pending_pipeline(pending)

            # Phase 2 complete — rerun to show final results cleanly
            st.rerun()

        else:
            # ── Normal render (no pending pipeline) ───────────────────────

            # Chat container
            chat_container = st.container(height=500, border=True)
            with chat_container:
                for msg in st.session_state.chat_messages:
                    with st.chat_message(
                        msg["role"],
                        avatar="🏥" if msg["role"] == "assistant" else "👤",
                    ):
                        st.markdown(msg["content"])

            # ── Image upload panel ────────────────────────────────────────
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
                        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        initial_state = build_lesion_state(
                            patient_id = st.session_state.current_patient_id,
                            image_path = temp_path,
                        )

                        st.session_state.running = True
                        with st.spinner("🤖 Analyzing skin lesion image…"):
                            _run_pipeline(
                                initial_state,
                                f"[Skin Lesion Analysis] Patient "
                                f"{st.session_state.current_patient_id} | "
                                f"file: {uploaded_file.name}",
                            )
                        st.rerun()

            # ── Chat input ────────────────────────────────────────────────
            user_input = st.chat_input(
                "Type your request here…",
                disabled=st.session_state.running or st.session_state.awaiting_image_upload,
            )

            # ── PHASE 1: user submits  (BUG 2 FIX) ───────────────────────
            # We do fast work here (routing = 1 embedding call, ~200ms):
            #   1. Append user message to chat_messages
            #   2. Append acknowledgment message based on route
            #   3. Store pending_pipeline + set running = True
            #   4. st.rerun() IMMEDIATELY — no pipeline yet
            #
            # The NEXT render (Phase 2 above) shows the messages first,
            # then executes the pipeline while the chat is already visible.
            if user_input:
                # Step 1 — echo user message
                st.session_state.chat_messages.append({
                    "role": "user", "content": user_input,
                })

                # Step 2 — fast semantic routing
                route_result      = route_request(user_input)
                proposed_category = route_result["category"]
                passed            = route_result["passed"]

                if not passed:
                    # Spam gate — no pipeline needed, reply immediately
                    st.session_state.chat_messages.append({
                        "role":    "assistant",
                        "content": (
                            "I'm sorry, I can only help with medical questions.\n\n"
                            "I can assist you with:\n"
                            "- 🩸 **Analyzing your blood test results**\n"
                            "- 🔬 **Analyzing a skin lesion image**\n"
                            "- 💊 **Answering general medical questions**\n\n"
                            "Please describe your health concern!"
                        ),
                    })
                    st.rerun()

                elif proposed_category == "blood_test_analysis":
                    # Check lab history before committing to pipeline
                    lab_history = get_patient_lab_history(
                        st.session_state.current_patient_id
                    )
                    if not lab_history:
                        st.session_state.chat_messages.append({
                            "role":    "assistant",
                            "content": (
                                "⚠️ I couldn't find any lab test results in your history. "
                                "Please contact your healthcare provider."
                            ),
                        })
                        st.rerun()
                    else:
                        st.session_state.chat_messages.append({
                            "role":    "assistant",
                            "content": "📋 Understood! Let me analyze your most recent blood test results…",
                        })
                        st.session_state.pending_pipeline = {
                            "user_input":   user_input,
                            "route_result": route_result,
                        }
                        st.session_state.running = True
                        st.rerun()  # ← Phase 1 ends here

                elif proposed_category == "image_lesion_analysis":
                    st.session_state.chat_messages.append({
                        "role":    "assistant",
                        "content": "🩺 Sure! Please upload a clear image of the lesion below so I can analyze it.",
                    })
                    st.session_state.awaiting_image_upload = True
                    st.rerun()

                elif proposed_category == "evidence_analyst":
                    st.session_state.chat_messages.append({
                        "role":    "assistant",
                        "content": "🔍 Let me look that up in our medical knowledge base…",
                    })
                    st.session_state.pending_pipeline = {
                        "user_input":   user_input,
                        "route_result": route_result,
                    }
                    st.session_state.running = True
                    st.rerun()  # ← Phase 1 ends here

                else:
                    st.session_state.chat_messages.append({
                        "role":    "assistant",
                        "content": (
                            "I'm not sure I understood your request. I can help you with:\n"
                            "- 🩸 **Analyzing your blood test results**\n"
                            "- 🔬 **Analyzing a skin lesion image**\n"
                            "- 💊 **Answering general medical questions**\n\n"
                            "Please rephrase your request!"
                        ),
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

                if st.session_state.last_state:
                    req_type = st.session_state.last_state.get("request_type")
                    if req_type == "image_lesion_analysis":
                        image_path     = st.session_state.last_state.get("image_path")
                        vision_results = st.session_state.last_state.get("vision_results")
                        if image_path and vision_results:
                            st.markdown("---")
                            st.markdown("**📊 Analysis Result**")
                            comparison_img = create_comparison_image(image_path, vision_results)
                            if comparison_img:
                                st.image(comparison_img, use_container_width=True)

        with st.expander("👤 Patient Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Patient ID", patient_data["id"])
                st.metric("Age", patient_data.get("age", "N/A"))
            with col2:
                st.metric("Sex", patient_data.get("sex", "N/A"))
                st.metric("Lab Tests", len(patient_data.get("lab_history", [])))

            if patient_data.get("chronic_conditions"):
                st.markdown("**Chronic Conditions:**")
                for condition in patient_data["chronic_conditions"]:
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