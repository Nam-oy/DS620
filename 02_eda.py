# 02_eda.py
import os, random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

CLEAN_CSV = "./data/ham_clean_binary.csv"
IMG_DIR = "./data/images"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(CLEAN_CSV)

# --- Plot 1: Original class distribution ---
plt.figure()
df["dx"].value_counts().plot(kind="bar")
plt.title("HAM10000 Original Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution_dx.png"))
plt.show()

# --- Plot 2: Binary risk distribution ---
plt.figure()
df["risk"].value_counts().plot(kind="bar")
plt.title("Binary Risk Distribution (0=Low, 1=High)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution_risk.png"))
plt.show()

# --- Show sample images ---
def show_samples(risk_value, n=5):
    subset = df[df["risk"] == risk_value]
    samples = subset.sample(n)

    plt.figure(figsize=(12, 3))
    for i, row in enumerate(samples.itertuples()):
        img_path = os.path.join(IMG_DIR, row.image_id + ".jpg")
        img = Image.open(img_path).convert("RGB")
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{row.dx}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"samples_risk_{risk_value}.png"))
    plt.show()

show_samples(0, n=5)
show_samples(1, n=5)

# --- Image size summary (optional descriptive stats) ---
sizes = []
for img_id in random.sample(list(df["image_id"]), 200):
    img = Image.open(os.path.join(IMG_DIR, img_id + ".jpg"))
    sizes.append(img.size)

w = [s[0] for s in sizes]
h = [s[1] for s in sizes]
print("Avg width:", sum(w)/len(w), "Avg height:", sum(h)/len(h))
