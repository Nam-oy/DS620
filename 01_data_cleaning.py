# 01_data_cleaning.py
import os
import pandas as pd

CSV_PATH = "./data/HAM10000_metadata.csv"
IMG_DIR = "./data/images"
CLEAN_CSV_OUT = "./data/ham_clean_binary.csv"

# Load metadata
df = pd.read_csv(CSV_PATH)

# --- Risk mapping (high-risk malignant vs low-risk benign) ---
HIGH_RISK = {"mel", "bcc", "akiec"}     # melanoma, basal cell, actinic keratoses/intraepithelial carcinoma
LOW_RISK  = {"nv", "bkl", "df", "vasc"} # benign classes

def risk_label(dx):
    return 1 if dx in HIGH_RISK else 0

df["risk"] = df["dx"].apply(risk_label)

# --- Remove duplicates if present ---
df = df.drop_duplicates(subset=["image_id"])

# --- Verify image files exist ---
missing = []
for img_id in df["image_id"]:
    path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(path):
        missing.append(img_id)

print("Missing images:", len(missing))
if missing:
    df = df[~df["image_id"].isin(missing)]

# --- Save cleaned file ---
df.to_csv(CLEAN_CSV_OUT, index=False)
print("Saved cleaned binary dataset:", CLEAN_CSV_OUT)
print(df["risk"].value_counts())
