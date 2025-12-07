import streamlit as st
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./outputs/models/best_resnet50.pth"

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.title("AI-Powered Skin Lesion Triage Assistant")

uploaded = st.file_uploader("Upload dermoscopic image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(1)).item()

    pred = "High Risk (Malignant)" if prob >= 0.5 else "Low Risk (Benign)"
    st.write("Prediction:", pred)
    st.write("Risk score:", round(prob, 3))
