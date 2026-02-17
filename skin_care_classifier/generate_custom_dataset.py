"""
generate_custom_dataset.py
==========================
Runs locally. Reads HAM10000 metadata + images and builds TWO balanced
binary datasets, each saved in its own folder:

  custom_dataset/        ← ~2,200 images (all mel + 1,100 nv)
  custom_dataset_600/    ← ~1,200 images (600 mel  + 600  nv)

Both are zipped and ready for upload to Google Colab.
"""

import zipfile
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent          # skin_care_classifier/
DATA_DIR  = BASE_DIR / "data"
META_CSV  = DATA_DIR / "HAM10000_metadata.csv"

IMAGE_DIRS = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_COL = "dx"
MEL_LABEL = "mel"   # Class 1 – High Urgency
NEV_LABEL = "nv"    # Class 0 – Low Urgency

KEEP_COLS = ["image_id", "image_path", "dx", "urgency_label",
             "dx_type", "age", "sex", "localization"]

dx_names = {
    "nv"   : "Melanocytic Nevi\n(nv)",
    "mel"  : "Melanoma\n(mel)",
    "bkl"  : "Benign Keratosis\n(bkl)",
    "bcc"  : "Basal Cell\nCarcinoma (bcc)",
    "akiec": "Actinic Keratosis\n(akiec)",
    "vasc" : "Vascular Lesion\n(vasc)",
    "df"   : "Dermatofibroma\n(df)",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load metadata (once)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata …")
meta = pd.read_csv(META_CSV)
print(f"  Total records : {len(meta):,}")
print(f"  Columns       : {list(meta.columns)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Build image-path lookup (once — scans both source dirs)
# ══════════════════════════════════════════════════════════════════════════════
print("Building image-path lookup …")
img_lookup: dict[str, Path] = {}
for img_dir in IMAGE_DIRS:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in img_dir.glob(ext):
            img_lookup[p.stem] = p
print(f"  Images found : {len(img_lookup):,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Helper – reusable dataset builder
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(mel_sample: int | None, nev_sample: int, out_dir: Path) -> None:
    """
    Sample mel_sample melanoma images (None = all) and nev_sample nevi images,
    copy them to out_dir/images/, save dataset_metadata.csv, plot a histogram,
    and zip everything into out_dir/<folder_name>.zip.
    """
    tag       = out_dir.name
    img_out   = out_dir / "images"
    out_csv   = out_dir / "dataset_metadata.csv"
    hist_path = out_dir / "class_distribution.png"
    zip_path  = out_dir / f"{tag}.zip"

    print("=" * 60)
    print(f"  Building dataset : {tag}")
    print("=" * 60)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    # ── Sample ────────────────────────────────────────────────────────────────
    mel_df = meta[meta[LABEL_COL] == MEL_LABEL].copy()
    nev_df = meta[meta[LABEL_COL] == NEV_LABEL].copy()

    if mel_sample is not None:
        mel_df = mel_df.sample(n=mel_sample, random_state=SEED)
        print(f"  Melanoma (mel) : sampled {mel_sample:,}")
    else:
        print(f"  Melanoma (mel) : keeping ALL {len(mel_df):,}")

    nev_df = nev_df.sample(n=nev_sample, random_state=SEED)
    print(f"  Nevi     (nv)  : sampled {nev_sample:,}")

    mel_df["urgency_label"] = 1
    nev_df["urgency_label"] = 0

    balanced = pd.concat([mel_df, nev_df], ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

    print(f"  Total          : {len(balanced):,} images\n")

    # ── Histogram ─────────────────────────────────────────────────────────────
    label_counts = meta[LABEL_COL].value_counts()
    colors_full  = ["#e74c3c" if d == "mel" else
                    "#3498db" if d == "nv"  else "#95a5a6"
                    for d in label_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"HAM10000 — {tag}", fontsize=14, fontweight="bold")

    ax = axes[0]
    bars = ax.bar([dx_names.get(d, d) for d in label_counts.index],
                  label_counts.values, color=colors_full,
                  edgecolor="white", linewidth=0.8)
    ax.set_title("Full Dataset (all classes)", fontweight="bold")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Diagnosis")
    ax.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, label_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30, str(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = axes[1]
    subset_counts = [len(mel_df), len(nev_df)]
    bars2 = ax2.bar(
        ["Melanoma\n(mel)\nClass 1 – High Urgency",
         "Nevi\n(nv)\nClass 0 – Low Urgency"],
        subset_counts,
        color=["#e74c3c", "#3498db"], edgecolor="white", linewidth=0.8
    )
    ax2.set_title(f"Balanced Subset ({sum(subset_counts):,} images)", fontweight="bold")
    ax2.set_ylabel("Number of images")
    ax2.set_xlabel("Binary Class")
    for bar, val in zip(bars2, subset_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 5, str(val),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Histogram saved → {hist_path}")

    # ── Copy images ───────────────────────────────────────────────────────────
    found, missing = 0, []
    resolved_paths = []

    for _, row in balanced.iterrows():
        src = img_lookup.get(row["image_id"])
        if src is None:
            missing.append(row["image_id"])
            resolved_paths.append(None)
        else:
            dst = img_out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            resolved_paths.append(str(dst))
            found += 1

    balanced["image_path"] = resolved_paths

    if missing:
        print(f"\n  ⚠  {len(missing)} image(s) not found:")
        for m in missing[:10]:
            print(f"     • {m}")
        if len(missing) > 10:
            print(f"     … and {len(missing) - 10} more")

    print(f"  Copied : {found:,} images → {img_out}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    balanced = balanced.dropna(subset=["image_path"]).reset_index(drop=True)
    keep = [c for c in KEEP_COLS if c in balanced.columns]
    balanced[keep].to_csv(out_csv, index=False)
    print(f"  CSV saved → {out_csv}")

    # ── Zip (ZIP_STORED — JPEGs are already compressed) ──────────────────────
    print("  Zipping …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        zf.write(out_csv, arcname=f"{tag}/dataset_metadata.csv")
        for img_file in img_out.iterdir():
            zf.write(img_file, arcname=f"{tag}/images/{img_file.name}")

    zip_mb = zip_path.stat().st_size / (1024 ** 2)
    print(f"  ✅  Zip saved → {zip_path}  ({zip_mb:.1f} MB)")
    print(f"\n  Unzip in Colab with:")
    print(f"  !unzip {tag}.zip -d /content/\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Generate BOTH datasets
# ══════════════════════════════════════════════════════════════════════════════

# Dataset A — ~2,200 images  (all mel + 1,100 nv)
# build_dataset(
#     mel_sample=None,
#     nev_sample=1100,
#     out_dir=BASE_DIR / "custom_dataset",
# )

# Dataset B — 1,200 images  (600 mel + 600 nv)
build_dataset(
    mel_sample=600,
    nev_sample=600,
    out_dir=BASE_DIR / "custom_dataset_600",
)

print("=" * 60)
print("  ALL DONE — both datasets generated and zipped.")
print("=" * 60)