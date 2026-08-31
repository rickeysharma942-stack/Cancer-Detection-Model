"""
Streamlit app: single-model cancer classifier (convnext_base headline model).

Downloads the model bundle from a GitHub Release URL on first run, caches it
locally, then loads it with timm and runs inference on an uploaded image.

Bundle format (from your training notebook):
    {"model_name", "cancer_type", "best_fold", "val_f1_macro", "class_names", "state_dict"}

Run:
    streamlit run app.py
"""

import io
import os
import pickle
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import streamlit as st
from torchvision.transforms import v2

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


MODEL_URL = "https://github.com/rickeysharma942-stack/Cancer-Detection-Model/releases/download/model-v1/obulisainaren_convnext_base_model.pkl"
LOCAL_MODEL_PATH = "obulisainaren_convnext_base_model.pkl"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CLAHE_Transform:
    """Matches the CLAHE preprocessing used during training."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if not HAS_CV2:
            return img
        try:
            np_img = np.array(img)
            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                return Image.fromarray(final)
        except Exception:
            pass
        return img


def build_val_transform(img_size=224):
    return v2.Compose([
        CLAHE_Transform(clip_limit=2.0),
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def robust_torch_load(path, device):
    """Load a checkpoint that may be a normal torch.save archive OR a plain
    pickle.dump() of a dict whose tensors are still tagged for CUDA storage
    (as some training notebooks produce when they save straight from GPU).

    Plain `torch.load(..., map_location=device)` can't rescue the second
    case on a CPU-only machine -- there's no torch archive for map_location
    to intercept, so it fails with "Invalid magic number" or a CUDA
    deserialize error. This falls back to a custom Unpickler that redirects
    torch's internal tensor-rebuild hook to load onto `device` instead.
    """
    try:
        with open(path, "rb") as f:
            return torch.load(f, map_location=device, weights_only=False)
    except RuntimeError:
        class _DeviceUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(io.BytesIO(b), map_location=device)
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _DeviceUnpickler(f).load()


@st.cache_resource(show_spinner="Downloading model (first run only)...")
def download_model(url, local_path):
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path


@st.cache_resource(show_spinner="Loading model...")
def load_model(local_path, device):
    bundle = robust_torch_load(local_path, device)

    model = timm.create_model(
        bundle["model_name"], pretrained=False, num_classes=len(bundle["class_names"])
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()

    return model, bundle["class_names"], bundle.get("val_f1_macro"), bundle.get("best_fold")


@torch.no_grad()
def predict(model, image, class_names, device):
    transform = build_val_transform(IMG_SIZE)
    x = transform(image).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
    return probs


# ------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------
st.title("🔬 Cancer Classifier")
st.caption("convnext_base — single best-fold model")

local_path = download_model(MODEL_URL, LOCAL_MODEL_PATH)
model, class_names, val_f1, best_fold = load_model(local_path, DEVICE)

st.success(f"Model loaded — best fold {best_fold}, val_f1_macro = {val_f1:.4f}" if val_f1 else "Model loaded")

uploaded_file = st.file_uploader("Upload a histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    with st.spinner("Running inference..."):
        probs = predict(model, image, class_names, DEVICE)

    top_idx = int(np.argmax(probs))
    st.subheader("Prediction")
    st.metric(label="Predicted class", value=class_names[top_idx], delta=f"{probs[top_idx]*100:.2f}% confidence")

    st.write("**All class probabilities:**")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        st.write(f"{class_names[i]}: {probs[i]*100:.2f}%")
        st.progress(float(probs[i]))

    st.caption(
        "⚠️ Research/educational tool only. Not a substitute for professional "
        "medical diagnosis. Always confirm with a qualified pathologist/oncologist."
    )
else:
    st.info("Upload an image to run the prediction.")
