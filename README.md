# 🧬 Cancer Detection using Machine Learning

A machine learning pipeline designed to predict medical diagnoses (e.g., Benign vs. Malignant) using patient diagnostic feature datasets.

---

## 📌 Project Overview
This project applies supervised machine learning classification algorithms to analyze patient clinical/diagnostic data and accurately detect cancerous tissue. It features automated data preprocessing, model training, evaluation metrics, and a minimal web interface for real-time predictions.

### Key Features
* **Exploratory Data Analysis (EDA):** Heatmaps, pair plots, and distribution visualizations.
* **Data Preprocessing:** Missing value handling, standard scaling, and feature selection.
* **Machine Learning Models:** Random Forest, Support Vector Machine (SVM), and Neural Networks.
* **Performance Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC curves.

---

## 📊 Dataset
This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**:
* **Features:** 30 numerical measurements computed from fine-needle aspirate (FNA) digitized images (e.g., radius, texture, perimeter, area, smoothness).
* **Target Label:** `1` (Malignant) or `0` (Benign).

---

## 📁 Repository Structure
```text
├── data/
│   └── dataset.csv           # Raw dataset file
├── notebooks/
│   └── eda_and_training.ipynb # Jupyter notebook for EDA & training
├── src/
│   ├── preprocess.py         # Scaling and transformation logic
│   ├── train.py              # Model training script
│   └── predict.py            # Inference script
├── models/
│   └── cancer_model.pkl      # Serialized trained model
├── requirements.txt          # Python dependencies
├── app.py                    # Inference API or Streamlit web app
└── README.md
