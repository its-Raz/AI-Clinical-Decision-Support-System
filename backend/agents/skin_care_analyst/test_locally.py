"""
Local test for the Skin Care Classifier Agent.

Usage:
    python test_local.py
    python test_local.py --image ISIC_0024323.jpg   # test a specific image

Expects this layout (relative to this file):
    data/
      test_split.csv          # columns: image,label  (or image_id, class_id)
      test_images/            # e.g. ISIC_0024323.jpg
      test_labels/            # e.g. ISIC_0024323.txt  (YOLO format)
"""

import os
import sys
import argparse
import random
import textwrap

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── path setup so we can import the agent without installing the package ──────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.skin_care_analyst.tools import classify_skin_lesion, _CLASS_META
from backend.agents.skin_care_analyst.agent import SkinCareAgent

# ── constants ─────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(__file__)
_DATA        = os.path.join(_HERE, "data")
_IMG_DIR     = os.path.join(_DATA, "test_images")
_LBL_DIR     = os.path.join(_DATA, "test_labels")
_CSV         = os.path.join(_DATA, "test_split.csv")

# Map class-id (int) → meta — mirrors _CLASS_META key order in tools.py
_ID_TO_META = {
    0: {"label": "Low Urgency",  "finding": "nevus (benign mole)"},
    1: {"label": "High Urgency", "finding": "possible melanoma"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_image_name(stem: str) -> str:
    """Given a bare stem (e.g. ISIC_0032879), find the actual file in test_images/."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = stem + ext
        if os.path.exists(os.path.join(_IMG_DIR, candidate)):
            return candidate
    return stem + ".jpg"  # fallback so error message is still useful


def pick_image(requested: str | None) -> str:
    """Return image filename (with extension) to test."""
    df = pd.read_csv(_CSV)

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    img_col = next(c for c in df.columns if "image" in c)

    if requested:
        stem = os.path.splitext(requested)[0]   # strip ext if user provided one
        return _resolve_image_name(stem)

    stem = str(random.choice(df[img_col].tolist()))
    return _resolve_image_name(stem)


def load_ground_truth(image_name: str) -> dict | None:
    """
    Parse YOLO label txt for the image.
    Format per line:  class_id  x_center  y_center  width  height   (normalised 0-1)
    Returns the first detection, or None if label file is missing / empty.
    """
    stem      = os.path.splitext(os.path.basename(image_name))[0]
    lbl_path  = os.path.join(_LBL_DIR, stem + ".txt")

    if not os.path.exists(lbl_path):
        print(f"⚠️  No label file found for {image_name}")
        return None

    with open(lbl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return None

    parts    = lines[0].split()
    class_id = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])
    meta     = _ID_TO_META.get(class_id, {"label": "Unknown", "finding": "unknown"})

    return {"class_id": class_id, "xc": xc, "yc": yc, "w": w, "h": h, **meta}


def yolo_to_pixel(box_norm: dict, img_w: int, img_h: int) -> tuple:
    """Convert normalised YOLO box → pixel (x1, y1, x2, y2)."""
    xc = box_norm["xc"] * img_w
    yc = box_norm["yc"] * img_h
    bw = box_norm["w"]  * img_w
    bh = box_norm["h"]  * img_h
    return xc - bw / 2, yc - bh / 2, bw, bh   # (x1, y1, w, h) for Rectangle


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(image_name: str, gt: dict | None, pred: dict) -> None:
    """Side-by-side: Ground Truth  |  Model Prediction."""
    img_path = os.path.join(_IMG_DIR, image_name)
    img_bgr  = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w     = img_rgb.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Skin Lesion Analysis — {image_name}", fontsize=13, fontweight="bold")

    # ── Left: Ground Truth ───────────────────────────────────────────────────
    axes[0].imshow(img_rgb)
    axes[0].set_title("Ground Truth", fontsize=11)
    axes[0].axis("off")

    if gt:
        x1, y1, bw, bh = yolo_to_pixel(gt, w, h)
        color = "red" if gt["label"] == "High Urgency" else "limegreen"
        rect  = patches.Rectangle((x1, y1), bw, bh,
                                   linewidth=2.5, edgecolor=color, facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 6, f"{gt['label']}\n({gt['finding']})",
                     color=color, fontsize=8.5,
                     bbox=dict(facecolor="black", alpha=0.55, pad=2))
    else:
        axes[0].text(0.5, 0.5, "No label file", transform=axes[0].transAxes,
                     ha="center", va="center", color="grey", fontsize=10)

    # ── Right: Model Prediction ───────────────────────────────────────────────
    axes[1].imshow(img_rgb)
    axes[1].set_title("Model Prediction", fontsize=11)
    axes[1].axis("off")

    if "error" not in pred:
        x1p, y1p, x2p, y2p = pred["bbox"]
        bw_p = x2p - x1p
        bh_p = y2p - y1p
        color_p = "red" if pred["label"] == "High Urgency" else "limegreen"
        rect_p  = patches.Rectangle((x1p, y1p), bw_p, bh_p,
                                     linewidth=2.5, edgecolor=color_p, facecolor="none")
        axes[1].add_patch(rect_p)
        axes[1].text(x1p, y1p - 6,
                     f"{pred['label']} ({pred['conf']*100:.1f}%)\n{pred['finding']}",
                     color=color_p, fontsize=8.5,
                     bbox=dict(facecolor="black", alpha=0.55, pad=2))
    else:
        axes[1].text(0.5, 0.5, f"Error:\n{pred['error']}",
                     transform=axes[1].transAxes,
                     ha="center", va="center", color="red", fontsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local test for Skin Care Analyst")
    parser.add_argument("--image", type=str, default=None,
                        help="Image filename to test (default: random from CSV)")
    args = parser.parse_args()

    # 1. Pick image
    image_name = pick_image(args.image)
    image_path = os.path.join(_IMG_DIR, image_name)
    print(f"\n🖼️  Testing image : {image_name}")

    # 2. Ground truth
    gt = load_ground_truth(image_name)
    if gt:
        print(f"📋 Ground truth  : {gt['label']} — {gt['finding']}")
    else:
        print("📋 Ground truth  : not available")

    # 3. Run YOLO tool directly (for the plot)
    pred = classify_skin_lesion.invoke({"image_path": image_path})
    if "error" in pred:
        print(f"❌ Model error   : {pred['error']}")
    else:
        print(f"🤖 Prediction    : {pred['label']} — {pred['finding']} "
              f"(conf: {pred['conf']*100:.1f}%)")

    # 4. Plot GT vs Prediction
    plot_results(image_name, gt, pred)

    # 5. Run full agent → patient report
    print("\n" + "─" * 55)
    print("📄 PATIENT REPORT (generated by agent):")
    print("─" * 55)

    state = {
        "request_type": "image_lesion_analysis",
        "patient_id":   "TEST_PATIENT",
        "image_path":   image_path,
        "lab_result":   None,
        "lab_insights": None,
        "vision_results": None,
        "messages":     [],
        "next_step":    "",
        "final_report": None,
    }

    agent        = SkinCareAgent()
    final_state  = agent.run(state)
    report       = final_state.get("final_report", "No report generated.")

    print(textwrap.fill(report, width=70))
    print("─" * 55 + "\n")


if __name__ == "__main__":
    main()