# 03_train_model.py (faster results, same model/report)

import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ✅ Use MPS on Mac if available, else CUDA, else CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)

# ✅ Faster CUDA convs
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

CLEAN_CSV = "./data/ham_clean_binary.csv"
IMG_DIR = "./data/images"
MODEL_OUT = "./outputs/models/best_resnet50.pth"
FIG_DIR = "./outputs/figures"
os.makedirs("./outputs/models", exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(CLEAN_CSV)

# --- Stratified split 70/15/15 ---
train_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["risk"], random_state=SEED
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1765, stratify=train_df["risk"], random_state=SEED
)

class HAMDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "risk"])
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ✅ Smaller image size speeds training a lot
IMG_SIZE = 160

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.05, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = HAMDataset(train_df, IMG_DIR, train_tf)
val_ds   = HAMDataset(val_df, IMG_DIR, val_tf)

# --- Weighted sampler for imbalance ---
counts = train_df["risk"].value_counts().to_dict()
weights = train_df["risk"].map(lambda x: 1.0/counts[x]).values
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ✅ Faster DataLoader
num_workers = 4 if DEVICE != "mps" else 0  # MPS can be unstable with workers > 0
batch_size = 64 if DEVICE != "cpu" else 32

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda"),
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda"),
)

# --- Transfer learning ResNet50 (keep same model) ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ Mixed precision for CUDA only
scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

def train_epoch():
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()

        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def val_epoch():
    model.eval()
    probs, labels = [], []
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p)
        labels.extend(y.numpy())

    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    return acc

# ✅ slightly fewer epochs = faster results
EPOCHS = 8
best_acc = 0
train_losses, val_accs = [], []

for ep in range(EPOCHS):
    loss = train_epoch()
    acc  = val_epoch()
    train_losses.append(loss)
    val_accs.append(acc)

    print(f"Epoch {ep+1}: loss={loss:.4f}, val_acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_OUT)
        print("✅ Saved best model")

# --- Plot training curves ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Training Curves (Faster)")
plt.savefig(os.path.join(FIG_DIR, "training_curves.png"))
plt.show()

print("Training complete. Best val acc:", best_acc)
