"""
frontend/app.py — Clinical Portal UI

Communicates exclusively with the backend via HTTP API:
    POST /api/execute  → run pipeline, get response + steps

Run with:
    streamlit run frontend/app.py

The API server must be running separately:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
EXECUTE_URL = f"{API_BASE}api/execute"
print(EXECUTE_URL)

# ─────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background: #f8fafc; }

    /* ── Header ── */
    .portal-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: white;
        padding: 22px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .portal-header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; color: white; }
    .portal-header p  { margin: 0; font-size: 0.85rem; opacity: 0.65; color: white; }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }

    /* ── Conversation bubbles ── */
    .bubble-user {
        background: #1e3a8a;
        color: white;
        border-radius: 14px 14px 4px 14px;
        padding: 12px 16px;
        margin: 6px 0 6px 40px;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .bubble-assistant {
        background: white;
        color: #1e293b;
        border-radius: 14px 14px 14px 4px;
        padding: 12px 16px;
        margin: 6px 40px 6px 0;
        font-size: 0.9rem;
        line-height: 1.6;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .bubble-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
        opacity: 0.6;
    }

    /* ── Step card ── */
    .step-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .step-index {
        display: inline-block;
        background: #0f172a;
        color: white;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        padding: 2px 7px;
        border-radius: 4px;
        margin-right: 8px;
    }
    .step-module {
        display: inline-block;
        background: #eff6ff;
        color: #1e40af;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        letter-spacing: 0.03em;
    }
    .step-field-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 10px 0 4px;
    }
    .step-field-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #334155;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 10px;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 160px;
        overflow-y: auto;
        line-height: 1.5;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        color: #94a3b8;
        padding: 60px 20px;
        font-size: 0.875rem;
    }
    .empty-state span { font-size: 2rem; display: block; margin-bottom: 10px; }

    /* ── Error banner ── */
    .error-banner {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.875rem;
        margin-top: 8px;
    }

    /* ── Textarea override ── */
    textarea {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
    }

    /* ── Run button ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 24px;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }
    div[data-testid="stButton"] > button:disabled { opacity: 0.45; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────

if "conversation" not in st.session_state:
    st.session_state.conversation = []   # list of {role, content}
if "last_steps" not in st.session_state:
    st.session_state.last_steps = []     # steps from the most recent run
if "running" not in st.session_state:
    st.session_state.running = False

# ─────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="portal-header">
    <div style="font-size:2rem;">🏥</div>
    <div>
        <h1>Clinical Portal</h1>
        <p>Autonomous Clinical AI &nbsp;·&nbsp; Blood Tests &nbsp;·&nbsp; Skin Analysis &nbsp;·&nbsp; Medical Q&amp;A</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Layout — two columns
# ─────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════
# LEFT — Conversation + Input
# ══════════════════════════════════════════════════════════════════════════

with left_col:

    # ── Conversation history ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    conv_container = st.container(height=420, border=True)
    with conv_container:
        if not st.session_state.conversation:
            st.markdown("""
            <div class="empty-state">
                <span>💬</span>
                Ask a medical question, request a blood test analysis,<br>
                or ask about a skin lesion.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.conversation:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="bubble-label" style="text-align:right;color:#1e3a8a;">You</div>
                    <div class="bubble-user">{msg["content"]}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bubble-label" style="color:#16a34a;">🏥 Clinical AI</div>
                    <div class="bubble-assistant">{msg["content"]}</div>
                    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Your prompt</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        label="prompt",
        label_visibility="collapsed",
        placeholder=(
            "Examples:\n"
            "• Can you explain my blood test results?\n"
            "• What is an allergy skin test?\n"
            "• Can you analyze this skin lesion?"
        ),
        height=110,
        disabled=st.session_state.running,
        key="prompt_input",
    )

    col_btn, col_clear = st.columns([3, 1])

    with col_btn:
        run_clicked = st.button(
            "▶  Run Agent" if not st.session_state.running else "⏳  Running…",
            disabled=st.session_state.running or not (user_input or "").strip(),
            use_container_width=True,
        )

    with col_clear:
        if st.button("Clear", use_container_width=True, disabled=st.session_state.running):
            st.session_state.conversation = []
            st.session_state.last_steps   = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# RIGHT — Steps trace
# ══════════════════════════════════════════════════════════════════════════

with right_col:
    st.markdown('<div class="section-label">Agent execution trace</div>', unsafe_allow_html=True)

    steps_container = st.container(height=540, border=True)
    with steps_container:
        if not st.session_state.last_steps:
            st.markdown("""
            <div class="empty-state">
                <span>⚙️</span>
                Step-by-step agent trace will appear here<br>after you run a prompt.
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, step in enumerate(st.session_state.last_steps, 1):
                module   = step.get("module", "—")
                prompt   = step.get("prompt", "")
                response = step.get("response", "")

                # Normalise prompt/response to display strings
                if isinstance(prompt, dict):
                    import json
                    prompt = json.dumps(prompt, indent=2)
                if isinstance(response, dict):
                    import json
                    response = json.dumps(response, indent=2)

                st.markdown(f"""
                <div class="step-card">
                    <div>
                        <span class="step-index">#{i:02d}</span>
                        <span class="step-module">{module}</span>
                    </div>
                    <div class="step-field-label">Prompt</div>
                    <div class="step-field-value">{prompt}</div>
                    <div class="step-field-label">Response</div>
                    <div class="step-field-value">{response}</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Pipeline execution  (triggered by Run Agent button)
# ─────────────────────────────────────────────────────────────────────────

if run_clicked and (user_input or "").strip():
    prompt_text = user_input.strip()

    # Append user message immediately
    st.session_state.conversation.append({"role": "user", "content": prompt_text})
    st.session_state.running    = True
    st.session_state.last_steps = []

    with st.spinner("🤖 Agent is running…"):
        try:
            resp = requests.post(
                EXECUTE_URL,
                json    = {"prompt": prompt_text},
                timeout = 180,
            )
            resp.raise_for_status()
            data = resp.json()

            agent_response = data.get("response") or ""
            steps          = data.get("steps", [])
            status         = data.get("status", "ok")
            error          = data.get("error")

            if status == "error" or error:
                st.session_state.conversation.append({
                    "role":    "assistant",
                    "content": f"⚠️ The agent returned an error:\n\n{error or 'Unknown error'}",
                })
            else:
                st.session_state.conversation.append({
                    "role":    "assistant",
                    "content": agent_response,
                })

            st.session_state.last_steps = steps

        except requests.exceptions.ConnectionError:
            st.session_state.conversation.append({
                "role":    "assistant",
                "content": (
                    f"⚠️ **Cannot connect to the API server.**\n\n"
                    f"Make sure the backend is running:\n"
                    f"```\nuvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload\n```\n"
                    f"Expected URL: `{EXECUTE_URL}`"
                ),
            })
        except requests.exceptions.Timeout:
            st.session_state.conversation.append({
                "role":    "assistant",
                "content": "⚠️ **Request timed out** (180 s). The pipeline may still be running — try again.",
            })
        except Exception as exc:
            st.session_state.conversation.append({
                "role":    "assistant",
                "content": f"⚠️ **Unexpected error:** {exc}",
            })

    st.session_state.running = False
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<hr style="border:none;border-top:1px solid #e2e8f0;margin:28px 0 10px;">
<div style="text-align:center;color:#94a3b8;font-size:0.78rem;">
    🔒 Secure Clinical Portal &nbsp;·&nbsp; AI-assisted &nbsp;·&nbsp; Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)