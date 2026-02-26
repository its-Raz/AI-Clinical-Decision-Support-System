"""
Tool: YOLO Skin Lesion Classifier
Wraps the YOLOv11 model as a LangChain tool.

Model class names:
  0: Low_Urgency_nv   (nv  = nevus, benign mole)
  1: High_Urgency_mel (mel = melanoma, malignant)
"""

import os
from typing import Dict, Any
from langchain_core.tools import tool

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "yolov11_skin_care_ob_det_classifier.pt"
)

# ── MOCKUP — replace with real model when ready ───────────────────────────
_MOCK_RESULT = {
    "bbox":      [100.0, 120.0, 340.0, 360.0],
    "raw_class": "Low_Urgency_nv",
    "label":     "Low Urgency",
    "finding":   "nevus (benign mole)",
    "conf":      0.9717,
}

_model = None  # reserved for real model


def _load_model():
    """Mockup — no model loaded."""
    pass


def preload_model():
    """Mockup — nothing to preload."""
    print("✅ [tools] YOLO mockup active — no model loaded")


@tool
def classify_skin_lesion(image_path: str) -> Dict[str, Any]:
    """
    Classify a skin lesion image using the YOLOv11 model.
    Returns bounding box, urgency label, clinical finding, and confidence score.

    Args:
        image_path: Path to the skin lesion image file.

    Returns:
        Dictionary with keys: bbox, raw_class, label, finding, conf, error (if any).
        - raw_class : original YOLO class name (e.g. "High_Urgency_mel")
        - label     : "High Urgency" or "Low Urgency"
        - finding   : short clinical descriptor (e.g. "possible melanoma")
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found at path: {image_path}"}

    # ── MOCKUP — returns hardcoded result ─────────────────────────────
    print(f"🟡 [classify_skin_lesion] MOCKUP — returning hardcoded result for: {image_path}")
    return _MOCK_RESULT