# make_report_figures.py
import os, random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import cv2

# ----------------------------
# 0. Config
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_PATH = "./data/HAM10000_metadata.csv"
IMG_DIR  = "./data/images"
CLEAN_CSV = "./data/ham_clean_binary.csv"

FIG_DIR = "./outputs/figures"
MODEL_DIR = "./outputs/models"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_resnet50.pth")

# ----------------------------
# 1. Load + Clean
# ----------------------------
df = pd.read_csv(CSV_PATH)

HIGH_RISK = {"mel", "bcc", "akiec"}
def risk_label(dx): return 1 if dx in HIGH_RISK else 0
df["risk"] = df["dx"].apply(risk_label)

df = df.drop_duplicates(subset=["image_id"])

missing = []
for img_id in df["image_id"]:
    if not os.path.exists(os.path.join(IMG_DIR, img_id + ".jpg")):
        missing.append(img_id)
if missing:
    df = df[~df["image_id"].isin(missing)]

df.to_csv(CLEAN_CSV, index=False)
print("Saved cleaned CSV:", CLEAN_CSV)

# ----------------------------
# 2. EDA Figures
# ----------------------------
# 2.1 original dx distribution
plt.figure()
df["dx"].value_counts().plot(kind="bar")
plt.title("HAM10000 Original Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution_dx.png"))
plt.close()

# 2.2 binary distribution
plt.figure()
df["risk"].value_counts().plot(kind="bar")
plt.title("Binary Risk Distribution (0=Low, 1=High)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution_risk.png"))
plt.close()

# 2.3 sample grids
def save_samples(risk_value, n=5):
    subset = df[df["risk"] == risk_value].sample(n, random_state=SEED)
    plt.figure(figsize=(12,3))
    for i, row in enumerate(subset.itertuples()):
        img = Image.open(os.path.join(IMG_DIR, row.image_id+".jpg")).convert("RGB")
        plt.subplot(1,n,i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(row.dx)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"samples_risk_{risk_value}.png"))
    plt.close()

save_samples(0, 5)
save_samples(1, 5)

print("✅ EDA figures saved")

# ----------------------------
# 3. Split + Dataset
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["risk"], random_state=SEED)
train_df, val_df  = train_test_split(train_df, test_size=0.1765, stratify=train_df["risk"], random_state=SEED)

class HAMDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "risk"])
        img = Image.open(os.path.join(IMG_DIR, img_id+".jpg")).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label, img_id

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = HAMDataset(train_df, train_tf)
val_ds   = HAMDataset(val_df, val_tf)
test_ds  = HAMDataset(test_df, val_tf)

# imbalance sampler
counts = train_df["risk"].value_counts().to_dict()
weights = train_df["risk"].map(lambda x: 1.0/counts[x]).values
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# ----------------------------
# 4. Model Train (ResNet50)
# ----------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch():
    model.train()
    total_loss = 0
    for x, y, _ in tqdm(train_loader):
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(train_loader.dataset)

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        p = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
        probs.extend(p); labels.extend(y.numpy())
    probs = np.array(probs); labels = np.array(labels)
    preds = (probs>=0.5).astype(int)
    acc = accuracy_score(labels, preds)
    return acc, probs, labels

EPOCHS=10
best_acc=0
train_losses=[]
val_accs=[]

for ep in range(EPOCHS):
    loss=train_epoch()
    val_acc,_,_=eval_epoch(val_loader)

    train_losses.append(loss)
    val_accs.append(val_acc)
    print(f"Epoch {ep+1}: loss={loss:.4f} val_acc={val_acc:.4f}")

    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(model.state_dict(), MODEL_PATH)

# training curve
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accs, label="Val Accuracy")
plt.title("Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"training_curves.png"))
plt.close()

print("✅ Training curve saved")

# ----------------------------
# 5. Test Evaluation Figures
# ----------------------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
test_acc, test_probs, test_labels = eval_epoch(test_loader)
test_preds=(test_probs>=0.5).astype(int)

cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Low Risk","High Risk"])
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"confusion_matrix.png"))
plt.close()

fpr, tpr, _ = roc_curve(test_labels, test_probs)
plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"roc_curve.png"))
plt.close()

print("✅ Confusion matrix + ROC curve saved")

# ----------------------------
# 6. Grad-CAM images
# ----------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model=model
        self.layer=layer
        self.grads=None
        self.acts=None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)
    def _save_acts(self,m,i,o): self.acts=o.detach()
    def _save_grads(self,m,gi,go): self.grads=go[0].detach()
    def generate(self,x):
        self.model.zero_grad()
        logits=self.model(x).squeeze(1)
        prob=torch.sigmoid(logits)
        prob.backward()
        w=self.grads.mean(dim=(2,3),keepdim=True)
        cam=(w*self.acts).sum(dim=1)
        cam=torch.relu(cam)[0].cpu().numpy()
        cam=cv2.resize(cam,(224,224))
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cam

cam_gen = GradCAM(model, model.layer4[-1])

def save_cam(img_id, outname):
    raw = Image.open(os.path.join(IMG_DIR,img_id+".jpg")).convert("RGB").resize((224,224))
    x = val_tf(raw).unsqueeze(0).to(DEVICE)
    cam = cam_gen.generate(x)

    img_np=np.array(raw)
    heatmap=cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay=heatmap*0.4+img_np

    plt.figure(figsize=(6,3))
    plt.subplot(1,3,1); plt.imshow(img_np); plt.axis("off"); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(cam,cmap="jet"); plt.axis("off"); plt.title("Grad-CAM")
    plt.subplot(1,3,3); plt.imshow(overlay.astype(np.uint8)); plt.axis("off"); plt.title("Overlay")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,outname))
    plt.close()

sample_ids = test_df.sample(3, random_state=SEED)["image_id"].tolist()
for i, sid in enumerate(sample_ids):
    save_cam(sid, f"gradcam_{i}.png")

print("✅ Grad-CAM figures saved")
print("All report figures are in:", FIG_DIR)
