"""
Streamlit app: multi-architecture cancer classification ensemble.

Loads every "*_best_fold*.pkl" checkpoint bundle it finds for a given
cancer_type in a checkpoint directory, groups them by architecture
(convnext_base, efficientnet_b0, densenet121, resnet50, ...), builds a
per-architecture fold-ensemble, then averages ACROSS architectures for a
final grand-ensemble prediction.

Each bundle is expected to be the format your training notebook saves:
    {"model_name", "cancer_type", "fold", "val_f1_macro", "class_names", "state_dict"}

Run:
    streamlit run app.py
"""

import os
import glob
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import streamlit as st
from torchvision.transforms import v2

# Optional CLAHE preprocessing to match training pipeline (falls back to no-op
# if opencv isn't installed in this environment).
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ------------------------------------------------------------------------
# CONFIG -- adjust these two lines for your environment
# ------------------------------------------------------------------------
CHECKPOINT_DIR = st.sidebar.text_input(
    "Checkpoint directory",
    value="/kaggle/input/models/rickeysharma/modle/pytorch/default/1",
    help="Folder containing your *_best_fold*.pkl bundles",
)
CANCER_NAME = st.sidebar.text_input("Cancer type prefix", value="obulisainaren")
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


@st.cache_resource(show_spinner="Loading checkpoints and building ensembles...")
def load_all_architectures(checkpoint_dir, cancer_name):
    """
    Scans checkpoint_dir for {cancer_name}_{model_name}_best_fold{N}.pkl files,
    groups them by architecture, and loads each fold's state_dict.

    Returns: dict {model_name: {"class_names": [...], "state_dicts": [...], "val_f1s": [...]}}
    """
    pattern = os.path.join(checkpoint_dir, f"{cancer_name}_*_best_fold*.pkl")
    paths = sorted(glob.glob(pattern))

    if not paths:
        return {}, f"No files matched pattern: {pattern}"

    by_model = defaultdict(lambda: {"class_names": None, "state_dicts": [], "val_f1s": [], "folds": []})

    for path in paths:
        with open(path, "rb") as f:
            bundle = torch.load(f, map_location="cpu", weights_only=False)

        model_name = bundle["model_name"]
        entry = by_model[model_name]

        if entry["class_names"] is None:
            entry["class_names"] = bundle["class_names"]
        elif entry["class_names"] != bundle["class_names"]:
            st.warning(f"class_names mismatch in {os.path.basename(path)} -- skipping this fold")
            continue

        entry["state_dicts"].append(bundle["state_dict"])
        entry["val_f1s"].append(bundle["val_f1_macro"])
        entry["folds"].append(bundle.get("fold", "?"))

    return dict(by_model), None


@st.cache_resource(show_spinner="Building models on device...")
def build_models(_by_model, device):
    """Instantiate actual torch models for every architecture/fold found."""
    built = {}
    for model_name, entry in _by_model.items():
        models = []
        for sd in entry["state_dicts"]:
            m = timm.create_model(model_name, pretrained=False, num_classes=len(entry["class_names"]))
            m.load_state_dict(sd)
            m.to(device).eval()
            models.append(m)
        built[model_name] = {"models": models, "class_names": entry["class_names"], "val_f1s": entry["val_f1s"]}
    return built


@torch.no_grad()
def predict_with_architecture(models, x):
    """Average softmax across all folds of one architecture."""
    probs = [F.softmax(m(x), dim=1) for m in models]
    return torch.stack(probs).mean(dim=0)


def run_inference(built, image, class_names, device):
    transform = build_val_transform(IMG_SIZE)
    x = transform(image).unsqueeze(0).to(device)

    per_arch_probs = {}
    for model_name, entry in built.items():
        per_arch_probs[model_name] = predict_with_architecture(entry["models"], x).cpu().numpy()[0]

    # Grand ensemble: average across architectures too
    grand_probs = np.mean(list(per_arch_probs.values()), axis=0)
    return per_arch_probs, grand_probs


# ------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------
st.title("🔬 Multi-Architecture Cancer Classifier")
st.caption("Ensembles convnext_base / efficientnet_b0 / densenet121 / resnet50 across all folds")

by_model, error = load_all_architectures(CHECKPOINT_DIR, CANCER_NAME)

if error:
    st.error(error)
    st.stop()

st.success(f"Found {len(by_model)} architecture(s): {', '.join(by_model.keys())}")
for model_name, entry in by_model.items():
    st.write(f"- **{model_name}**: {len(entry['state_dicts'])} fold(s), "
             f"val_f1_macro = {[f'{f:.4f}' for f in entry['val_f1s']]}")

built = build_models(by_model, DEVICE)
class_names = next(iter(by_model.values()))["class_names"]

uploaded_file = st.file_uploader("Upload a histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    with st.spinner("Running inference across all architectures..."):
        per_arch_probs, grand_probs = run_inference(built, image, class_names, DEVICE)

    # ---- Final grand-ensemble result ----
    top_idx = int(np.argmax(grand_probs))
    st.subheader("🏆 Final Ensemble Prediction")
    st.metric(label="Predicted class", value=class_names[top_idx], delta=f"{grand_probs[top_idx]*100:.2f}% confidence")

    st.write("**All class probabilities (grand ensemble):**")
    sorted_idx = np.argsort(grand_probs)[::-1]
    for i in sorted_idx:
        st.write(f"{class_names[i]}: {grand_probs[i]*100:.2f}%")
        st.progress(float(grand_probs[i]))

    # ---- Per-architecture breakdown ----
    with st.expander("See per-architecture breakdown"):
        for model_name, probs in per_arch_probs.items():
            idx = int(np.argmax(probs))
            st.write(f"**{model_name}** -> {class_names[idx]} ({probs[idx]*100:.2f}%)")

    st.caption(
        "⚠️ Research/educational tool only. Not a substitute for professional "
        "medical diagnosis. Always confirm with a qualified pathologist/oncologist."
    )
else:
    st.info("Upload an image to run the ensemble prediction.")
