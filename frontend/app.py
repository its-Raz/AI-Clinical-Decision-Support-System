"""
frontend/app.py — Streamlit Demo UI

Split-screen layout:
  Left  → Patient View   : clean chat interface
  Right → System Trace   : developer view showing internal messages

Supports TWO workflows:
  1. Blood Test Analysis  (button: "Simulate New Lab Result")
  2. Skin Care Analysis   (button: "Skin Care Analysis" → image upload)

Run with:
    streamlit run frontend/app.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from backend.agents.graph import build_system, trigger_p002_lab_result

# ─────────────────────────────────────────────────────────────────────────
# Ground Truth & Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "backend", "agents", "skin_care_analyst", "data"
)
_LABEL_DIR = os.path.join(_DATA_DIR, "test_labels")

_CLASS_META = {
    0: {"label": "Low Urgency",  "color": (0, 255, 0)},    # green
    1: {"label": "High Urgency", "color": (255, 0, 0)},    # red
}


def _load_ground_truth(image_path: str) -> dict | None:
    """Load ground truth YOLO label for the image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(_LABEL_DIR, stem + ".txt")

    print(f"[GT Loader] image_path: {image_path}")
    print(f"[GT Loader] stem: {stem}")
    print(f"[GT Loader] label_path: {label_path}")
    print(f"[GT Loader] exists: {os.path.exists(label_path)}")

    if not os.path.exists(label_path):
        st.warning(f"⚠️ Ground truth not found: {stem}.txt")
        return None

    try:
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            return None

        parts = lines[0].split()
        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])

        meta = _CLASS_META.get(class_id, {"label": "Unknown", "color": (128, 128, 128)})

        print(f"[GT Loader] ✅ Loaded: class={class_id}, label={meta['label']}")

        return {
            "class_id": class_id,
            "xc": xc, "yc": yc, "w": w, "h": h,
            "label": meta["label"],
            "color": meta["color"],
        }
    except Exception as e:
        st.warning(f"Failed to load ground truth: {e}")
        print(f"[GT Loader] ❌ Error: {e}")
        return None


def _draw_bbox_pil(image_pil: Image.Image, bbox: dict, label: str, color: tuple,
                   line_width: int = 3) -> Image.Image:
    """Draw a bounding box on PIL image."""
    draw = ImageDraw.Draw(image_pil)
    img_w, img_h = image_pil.size

    # Convert normalized YOLO to pixel coords if needed
    if "xc" in bbox:  # Ground truth format
        xc = bbox["xc"] * img_w
        yc = bbox["yc"] * img_h
        bw = bbox["w"] * img_w
        bh = bbox["h"] * img_h
        x1, y1 = xc - bw/2, yc - bh/2
        x2, y2 = xc + bw/2, yc + bh/2
    else:  # Prediction format [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    # Draw label
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Background for text
    text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)

    return image_pil


def _create_comparison_image(image_path: str, prediction: dict) -> Image.Image | None:
    """Create single image with both GT and Prediction boxes overlaid."""
    try:
        print(f"\n[Comparison Image] Creating visualization...")
        print(f"[Comparison Image] image_path: {image_path}")
        print(f"[Comparison Image] prediction: {prediction}")

        # Load image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        print(f"[Comparison Image] Image size: {img_w}x{img_h}")

        try:
            font_label = ImageFont.truetype("arial.ttf", 18)
            font_legend = ImageFont.truetype("arial.ttf", 14)
        except:
            font_label = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        boxes_drawn = []

        # ── Draw Ground Truth (GREEN) ─────────────────────────────────
        gt = _load_ground_truth(image_path)
        if gt:
            print(f"[Comparison Image] Drawing GT box: {gt['label']}")
            xc = gt["xc"] * img_w
            yc = gt["yc"] * img_h
            bw = gt["w"] * img_w
            bh = gt["h"] * img_h
            x1, y1 = xc - bw/2, yc - bh/2
            x2, y2 = xc + bw/2, yc + bh/2

            print(f"[Comparison Image] GT bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

            gt_color = (0, 255, 0)  # GREEN
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=4)

            # Label at top-left of box
            label_text = f"GT: {gt['label']}"
            text_bbox = draw.textbbox((x1, y1 - 28), label_text, font=font_label)
            draw.rectangle(text_bbox, fill=gt_color)
            draw.text((x1, y1 - 28), label_text, fill=(0, 0, 0), font=font_label)

            boxes_drawn.append(("Ground Truth", gt_color))
        else:
            print(f"[Comparison Image] ⚠️ No GT found")

        # ── Draw Prediction (YELLOW/ORANGE) ───────────────────────────
        if prediction and "bbox" in prediction:
            print(f"[Comparison Image] Drawing Prediction box: {prediction.get('label')}")
            x1p, y1p, x2p, y2p = prediction["bbox"]
            pred_label = prediction.get("label", "Unknown")
            pred_conf = prediction.get("conf", 0)

            print(f"[Comparison Image] Pred bbox: ({x1p:.1f}, {y1p:.1f}, {x2p:.1f}, {y2p:.1f})")

            pred_color = (255, 165, 0)  # ORANGE
            draw.rectangle([x1p, y1p, x2p, y2p], outline=pred_color, width=4)

            # Label at bottom-right of box (to avoid overlap with GT label)
            label_text = f"Pred: {pred_label} ({pred_conf:.1%})"
            text_bbox = draw.textbbox((x1p, y2p + 4), label_text, font=font_label)
            draw.rectangle(text_bbox, fill=pred_color)
            draw.text((x1p, y2p + 4), label_text, fill=(0, 0, 0), font=font_label)

            boxes_drawn.append(("Model Prediction", pred_color))
        else:
            print(f"[Comparison Image] ⚠️ No prediction bbox")

        print(f"[Comparison Image] Total boxes drawn: {len(boxes_drawn)}")

        # ── Legend in top-right corner ────────────────────────────────
        if boxes_drawn:
            legend_x = img_w - 200
            legend_y = 10

            # Background for legend
            legend_height = 20 + len(boxes_drawn) * 25
            draw.rectangle(
                [legend_x - 10, legend_y - 5, img_w - 10, legend_y + legend_height],
                fill=(40, 40, 40, 220)
            )

            for i, (name, color) in enumerate(boxes_drawn):
                y_offset = legend_y + i * 25
                # Color square
                draw.rectangle(
                    [legend_x, y_offset, legend_x + 15, y_offset + 15],
                    fill=color
                )
                # Text
                draw.text(
                    (legend_x + 20, y_offset),
                    name,
                    fill=(255, 255, 255),
                    font=font_legend
                )

        return img

    except Exception as e:
        st.error(f"Failed to create comparison image: {e}")
        return None

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
    st.session_state.chat_messages = []

if "trace_messages" not in st.session_state:
    st.session_state.trace_messages = []

if "running" not in st.session_state:
    st.session_state.running = False

if "last_state" not in st.session_state:
    st.session_state.last_state = None

if "show_image_uploader" not in st.session_state:
    st.session_state.show_image_uploader = False

if "initialized" not in st.session_state:
    # Add initial greeting
    st.session_state.chat_messages = [{
        "role": "assistant",
        "content": (
            "Hi, I'm your medical assistant. I can review your lab test results "
            "as they come in, or you can provide an image of a skin lesion for "
            "a preliminary analysis."
        ),
    }]
    st.session_state.initialized = True

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
    "system":    "#2d3748",
    "assistant": "#1a365d",
    "tool":      "#2d3a1e",
    "user":      "#3d2020",
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
    """Separate trace from patient-facing messages."""
    patient_msgs = [m for m in all_messages if m.get("role") == "assistant"]
    trace_msgs   = all_messages
    return patient_msgs, trace_msgs


def _run_pipeline(initial_state: dict, trigger_description: str):
    """Common pipeline execution logic for both workflows."""
    st.session_state.running = True

    # Keep existing chat messages, just add new results
    existing_chat = st.session_state.chat_messages.copy()
    st.session_state.trace_messages = []
    st.session_state.last_state = None

    with st.spinner("🤖 Agents running — this may take 30–60 seconds…"):
        try:
            # Log the trigger event
            st.session_state.trace_messages.append({
                "role":    "system",
                "content": trigger_description,
            })

            system = build_system()
            final_state = system.run(initial_state)

            # Split messages
            all_msgs = final_state.get("messages", [])
            patient_msgs, trace_msgs = _split_messages(all_msgs)

            # Preserve existing + add new patient messages
            st.session_state.chat_messages = existing_chat + patient_msgs
            st.session_state.trace_messages = (
                st.session_state.trace_messages + trace_msgs
            )
            st.session_state.last_state = final_state

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
        finally:
            st.session_state.running = False


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

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

with ctrl_col1:
    blood_test_btn = st.button(
        "🧪 Simulate New Lab Result",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
        help="Loads P002's most recent lab batch.",
    )

with ctrl_col2:
    skin_care_btn = st.button(
        "🩺 Skin Care Analysis",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state.running,
        help="Upload a skin lesion image for preliminary analysis.",
    )

with ctrl_col3:
    clear_btn = st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
        help="Reset the conversation.",
    )

# ── Button actions ────────────────────────────────────────────────────────

if clear_btn:
    st.session_state.chat_messages = [{
        "role": "assistant",
        "content": (
            "Hi, I'm your medical assistant. I can review your lab test results "
            "as they come in, or you can provide an image of a skin lesion for "
            "a preliminary analysis."
        ),
    }]
    st.session_state.trace_messages = []
    st.session_state.last_state = None
    st.session_state.show_image_uploader = False
    st.rerun()

if skin_care_btn:
    st.session_state.show_image_uploader = not st.session_state.show_image_uploader
    st.rerun()

if blood_test_btn:
    initial_state = trigger_p002_lab_result()
    _run_pipeline(
        initial_state,
        f"[Trigger] New lab result event for patient {initial_state['patient_id']}. "
        f"{len(initial_state['lab_result'])} metrics in batch."
    )
    st.rerun()

# ── Image uploader (conditional) ──────────────────────────────────────────

if st.session_state.show_image_uploader:
    st.markdown("### 📤 Upload Skin Lesion Image")

    st.info(
        "💡 **For ground truth comparison**: Upload images from "
        "`backend/agents/skin_care_analyst/data/test_images/` directory. "
        "The system will automatically load the corresponding label file and display both "
        "ground truth (green) and prediction (orange) boxes overlaid on the image.",
        icon="ℹ️"
    )

    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of the skin lesion.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        analyze_btn = st.button(
            "▶️ Analyze Image",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.running,
        )

        if analyze_btn:
            # Save uploaded file to temp directory with ORIGINAL filename
            # (critical: GT loader needs the original name to find the .txt label)
            temp_path = os.path.join(
                tempfile.gettempdir(),
                uploaded_file.name  # ← preserve original name like ISIC_0032879.jpg
            )

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Build state
            initial_state = {
                "request_type":   "image_lesion_analysis",
                "patient_id":     "DEMO_PATIENT",  # or extract from session
                "lab_result":     None,
                "lab_insights":   None,
                "image_path":     temp_path,
                "vision_results": None,
                "vision_insights": None,
                "messages":       [],
                "next_step":      "",
                "final_report":   None,
            }

            _run_pipeline(
                initial_state,
                f"[Trigger] Skin lesion analysis for patient {initial_state['patient_id']}. "
                f"Image: {uploaded_file.name}"
            )

            st.session_state.show_image_uploader = False  # Hide uploader after analysis
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

    chat_container = st.container(height=600, border=True)

    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown(
                "<p style='color:#4a5568;text-align:center;margin-top:180px'>"
                "Waiting for analysis…</p>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_messages:
                with st.chat_message("assistant", avatar="🏥"):
                    st.markdown(msg["content"])

    st.markdown(
        "<p style='color:#4a5568;font-size:0.78rem;margin-top:4px'>"
        "💡 Follow-up chat coming in next iteration.</p>",
        unsafe_allow_html=True,
    )
    st.chat_input("Ask a follow-up question…", disabled=True, key="patient_input")


# ── RIGHT — System Trace ──────────────────────────────────────────────────
with right_col:
    st.markdown(
        "<h3 style='color:#68d391;margin-bottom:8px'>⚙️ System Trace</h3>",
        unsafe_allow_html=True,
    )

    trace_container = st.container(height=600, border=True)

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

            # ── Visualization: Ground Truth vs Prediction ─────────────
            if st.session_state.last_state:
                req_type = st.session_state.last_state.get("request_type")

                if req_type == "image_lesion_analysis":
                    image_path = st.session_state.last_state.get("image_path")
                    vision_results = st.session_state.last_state.get("vision_results")

                    if image_path and vision_results:
                        st.markdown("---")
                        st.markdown(
                            "<p style='color:#68d391;font-weight:bold;margin:8px 0'>📊 Visual Analysis</p>",
                            unsafe_allow_html=True
                        )

                        comparison_img = _create_comparison_image(image_path, vision_results)
                        if comparison_img:
                            st.image(
                                comparison_img,
                                use_container_width=True,
                                caption="Green = Ground Truth | Orange = Model Prediction"
                            )

    # ── State inspector ───────────────────────────────────────────────────
    if st.session_state.last_state:
        with st.expander("🔍 Raw AgentState (debug)", expanded=False):
            display_state = {
                k: v for k, v in st.session_state.last_state.items()
                if k != "messages"
            }
            st.json(display_state)

# ─────────────────────────────────────────────────────────────────────────
# Status bar
# ─────────────────────────────────────────────────────────────────────────

if st.session_state.last_state:
    state      = st.session_state.last_state
    req_type   = state.get("request_type", "N/A")
    n_trace    = len(st.session_state.trace_messages)
    has_report = bool(state.get("final_report"))

    # Get relevant metric based on request type
    if req_type == "blood_test_analysis":
        metric_label = "Metrics"
        metric_value = len(state.get("lab_result") or [])
    elif req_type == "image_lesion_analysis":
        metric_label = "YOLO Conf"
        vision_res = state.get("vision_results") or {}
        conf_val = vision_res.get("conf", 0)
        metric_value = f"{conf_val:.1%}" if conf_val else "N/A"
    else:
        metric_label = "Data"
        metric_value = "N/A"

    st.markdown(
        f"""
        <div style="
            background:#1a202c;border-radius:6px;
            padding:8px 16px;margin-top:8px;
            display:flex;gap:24px;font-size:0.8rem;color:#718096;
        ">
            <span>Patient: <b style='color:#e2e8f0'>{state.get('patient_id')}</b></span>
            <span>Type: <b style='color:#e2e8f0'>{req_type}</b></span>
            <span>{metric_label}: <b style='color:#e2e8f0'>{metric_value}</b></span>
            <span>Trace events: <b style='color:#e2e8f0'>{n_trace}</b></span>
            <span>Report: <b style='color:{"#68d391" if has_report else "#fc8181"}'>
                {"✓ Ready" if has_report else "✗ Pending"}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )