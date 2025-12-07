# 04_evaluate_and_gradcam.py
import os, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
from PIL import Image
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEAN_CSV = "./data/ham_clean_binary.csv"
IMG_DIR = "./data/images"
MODEL_PATH = "./outputs/models/best_resnet50.pth"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(CLEAN_CSV)

# test split must match train script
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["risk"], random_state=42)

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

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
        return img, label, img_path

test_ds = HAMDataset(test_df, IMG_DIR, val_tf)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---- evaluation ----
all_probs, all_labels = [], []

with torch.no_grad():
    for x, y, _ in test_loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
preds = (all_probs >= 0.5).astype(int)

acc  = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds)
rec  = recall_score(all_labels, preds)
f1   = f1_score(all_labels, preds)
auc  = roc_auc_score(all_labels, all_probs)

print("=== TEST RESULTS ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("ROC-AUC  :", auc)

cm = confusion_matrix(all_labels, preds)
print("Confusion Matrix:\n", cm)

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"))
plt.show()


# ---- Grad-CAM ----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.activations = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x):
        self.model.zero_grad()
        logits = self.model(x).squeeze(1)
        prob = torch.sigmoid(logits)
        prob.backward()

        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (w * self.activations).sum(dim=1)
        cam = torch.relu(cam).cpu().numpy()
        cam = cam[0]
        cam = cv2.resize(cam, (224,224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

cam_gen = GradCAM(model, model.layer4[-1])

def save_gradcam_example(img_path, out_name):
    raw = Image.open(img_path).convert("RGB").resize((224,224))
    x = val_tf(raw).unsqueeze(0).to(DEVICE)

    cam = cam_gen.generate(x)
    img_np = np.array(raw)

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_np

    plt.figure(figsize=(6,3))
    plt.subplot(1,3,1); plt.imshow(img_np); plt.axis("off"); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(cam, cmap="jet"); plt.axis("off"); plt.title("Grad-CAM")
    plt.subplot(1,3,3); plt.imshow(overlay.astype(np.uint8)); plt.axis("off"); plt.title("Overlay")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, out_name))
    plt.show()

# Save 3 examples for report
sample_paths = test_df.sample(3, random_state=1)["image_id"].tolist()
for i, img_id in enumerate(sample_paths):
    save_gradcam_example(os.path.join(IMG_DIR, img_id+".jpg"), f"gradcam_{i}.png")
