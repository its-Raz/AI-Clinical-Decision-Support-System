"""
frontend/app.py — Streamlit Demo UI

Split-screen layout:
  Left  → Patient View   : clean chat interface showing the final report
  Right → System Trace   : developer view showing every internal message

Run with:
    streamlit run frontend/app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from backend.agents.graph import build_system, trigger_p002_lab_result

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Autonomous Clinical System",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []   # patient-facing messages

if "trace_messages" not in st.session_state:
    st.session_state.trace_messages = []  # full internal trace

if "running" not in st.session_state:
    st.session_state.running = False

if "last_state" not in st.session_state:
    st.session_state.last_state = None

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

ROLE_ICON = {
    "system":    "⚙️",
    "assistant": "🏥",
    "tool":      "🔧",
    "user":      "👤",
}

ROLE_COLOR = {
    "system":    "#2d3748",   # dark slate
    "assistant": "#1a365d",   # dark blue
    "tool":      "#2d3a1e",   # dark green
    "user":      "#3d2020",   # dark red
}

def _render_trace_message(msg: dict, index: int):
    """Render a single internal trace message as a styled card."""
    role    = msg.get("role", "system")
    content = msg.get("content", "")
    icon    = ROLE_ICON.get(role, "•")
    color   = ROLE_COLOR.get(role, "#2d3748")

    st.markdown(
        f"""
        <div style="
            background:{color};
            border-left: 3px solid {'#63b3ed' if role=='system' else '#68d391' if role=='tool' else '#fc8181'};
            padding:8px 12px;
            border-radius:4px;
            margin-bottom:6px;
            font-size:0.82rem;
            color:#e2e8f0;
        ">
            <span style="opacity:0.6;font-size:0.75rem">#{index} {icon} {role.upper()}</span><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _split_messages(all_messages: list):
    """
    Separate internal trace messages from patient-facing assistant messages.
    Returns (patient_msgs, trace_msgs).
    """
    patient_msgs = [m for m in all_messages if m.get("role") == "assistant"]
    trace_msgs   = all_messages  # show everything in the trace panel
    return patient_msgs, trace_msgs


# ─────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='text-align:center;color:#63b3ed;font-size:1.8rem;margin-bottom:0'>
        🏥 Autonomous Clinical System
    </h1>
    <p style='text-align:center;color:#718096;margin-top:4px;font-size:0.9rem'>
        Multi-Agent Medical Triage &nbsp;|&nbsp; Demo
    </p>
    <hr style='border-color:#2d3748;margin:12px 0'>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────
# Control Panel
# ─────────────────────────────────────────────────────────────────────────

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 2])

with ctrl_col2:
    trigger_btn = st.button(
        "🧪 Simulate New Lab Result",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
        help="Loads P002's most recent lab batch and runs the full agent pipeline.",
    )

# ─────────────────────────────────────────────────────────────────────────
# Run pipeline on button click
# ─────────────────────────────────────────────────────────────────────────

if trigger_btn:
    st.session_state.running       = True
    st.session_state.chat_messages = []
    st.session_state.trace_messages = []
    st.session_state.last_state    = None

    with st.spinner("🤖 Agents running — this may take 30–60 seconds…"):
        try:
            initial_state = trigger_p002_lab_result()

            # Show the trigger event as first trace entry
            st.session_state.trace_messages.append({
                "role":    "system",
                "content": (
                    f"[Trigger] New lab result event for patient "
                    f"{initial_state['patient_id']}. "
                    f"{len(initial_state['lab_result'])} metrics in batch."
                ),
            })

            system = build_system()
            final_state = system.run(initial_state)

            # Split messages
            all_msgs = final_state.get("messages", [])
            patient_msgs, trace_msgs = _split_messages(all_msgs)

            st.session_state.chat_messages  = patient_msgs
            st.session_state.trace_messages = (
                st.session_state.trace_messages + trace_msgs
            )
            st.session_state.last_state     = final_state

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
        finally:
            st.session_state.running = False

    st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# Split-screen layout
# ─────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="medium")

# ── LEFT — Patient View ───────────────────────────────────────────────────
with left_col:
    st.markdown(
        "<h3 style='color:#63b3ed;margin-bottom:8px'>👤 Patient View</h3>",
        unsafe_allow_html=True,
    )

    chat_container = st.container(height=520, border=True)

    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown(
                "<p style='color:#4a5568;text-align:center;margin-top:180px'>"
                "Waiting for new lab results…<br>"
                "<small>Press the button above to simulate.</small></p>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_messages:
                with st.chat_message("assistant", avatar="🏥"):
                    st.markdown(msg["content"])

    # ── Patient chat input (for follow-up Q&A — future iteration) ────────
    st.markdown(
        "<p style='color:#4a5568;font-size:0.78rem;margin-top:4px'>"
        "💡 Patient follow-up chat coming in next iteration.</p>",
        unsafe_allow_html=True,
    )
    st.chat_input("Ask a follow-up question…", disabled=True, key="patient_input")


# ── RIGHT — System Trace ──────────────────────────────────────────────────
with right_col:
    st.markdown(
        "<h3 style='color:#68d391;margin-bottom:8px'>⚙️ System Trace</h3>",
        unsafe_allow_html=True,
    )

    trace_container = st.container(height=520, border=True)

    with trace_container:
        if not st.session_state.trace_messages:
            st.markdown(
                "<p style='color:#4a5568;text-align:center;margin-top:180px'>"
                "No activity yet.</p>",
                unsafe_allow_html=True,
            )
        else:
            for i, msg in enumerate(st.session_state.trace_messages, 1):
                _render_trace_message(msg, i)

    # ── State inspector ───────────────────────────────────────────────────
    if st.session_state.last_state:
        with st.expander("🔍 Raw AgentState (debug)", expanded=False):
            display_state = {
                k: v for k, v in st.session_state.last_state.items()
                if k != "messages"           # messages shown in trace above
            }
            st.json(display_state)

# ─────────────────────────────────────────────────────────────────────────
# Status bar
# ─────────────────────────────────────────────────────────────────────────

if st.session_state.last_state:
    state      = st.session_state.last_state
    n_trace    = len(st.session_state.trace_messages)
    has_report = bool(state.get("final_report"))

    st.markdown(
        f"""
        <div style="
            background:#1a202c;border-radius:6px;
            padding:8px 16px;margin-top:8px;
            display:flex;gap:24px;font-size:0.8rem;color:#718096;
        ">
            <span>Patient: <b style='color:#e2e8f0'>{state.get('patient_id')}</b></span>
            <span>Metrics: <b style='color:#e2e8f0'>{len(state.get('lab_result') or [])}</b></span>
            <span>Trace events: <b style='color:#e2e8f0'>{n_trace}</b></span>
            <span>Report: <b style='color:{"#68d391" if has_report else "#fc8181"}'>
                {"✓ Ready" if has_report else "✗ Pending"}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
