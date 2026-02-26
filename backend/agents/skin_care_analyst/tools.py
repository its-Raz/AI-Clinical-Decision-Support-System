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
_model = None  # Lazy-loaded singleton

# Map raw YOLO class names → human-readable label + clinical hint
_CLASS_META = {
    "Low_Urgency_nv":   {"label": "Low Urgency",  "finding": "nevus (benign mole)"},
    "High_Urgency_mel": {"label": "High Urgency",  "finding": "possible melanoma"},
}


def _load_model():
    """Load YOLO model once and cache it."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(_MODEL_PATH)
    return _model


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

    try:
        model = _load_model()
        results = model(image_path)[0]

        if not results.boxes:
            return {"error": "No lesion detected in the image."}

        # Take the highest-confidence detection
        box      = results.boxes[0]
        bbox     = box.xyxy[0].tolist()           # [x1, y1, x2, y2]
        conf     = round(float(box.conf[0]), 4)
        class_id = int(box.cls[0])
        raw_class = results.names[class_id]       # "Low_Urgency_nv" or "High_Urgency_mel"

        meta = _CLASS_META.get(raw_class, {"label": raw_class, "finding": "unknown"})

        return {
            "bbox":      bbox,
            "raw_class": raw_class,
            "label":     meta["label"],
            "finding":   meta["finding"],
            "conf":      conf,
        }

    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}


def preload_model():
    """Explicitly load the YOLO model at startup so first request is not slow."""
    _load_model()
    print("✅ [tools] YOLO model loaded and ready")