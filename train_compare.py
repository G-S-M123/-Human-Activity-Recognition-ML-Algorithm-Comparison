"""
train_and_compare.py
---------------------
Core logic for ML algorithm comparison on the HARTH dataset.
Updated for HARTH structure:
- Drop timestamp
- Target = label
- Save label mapping for UI
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------------------------------------------------
# 1️⃣ PATH SETUP
# -------------------------------------------------------------
DATA_PATH = r"F:\10_study\Harth_Algo_compare\har70plus\507.csv"       
MODEL_DIR = "./models"
RESULTS_PATH = "./results/comparison_results.csv"
LABEL_MAP_PATH = "./models/label_mapping.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# -------------------------------------------------------------
# 2️⃣ LOAD + PREPROCESS DATA
# -------------------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Drop timestamp (not used for training)
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for Streamlit app
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# -------------------------------------------------------------
# 3️⃣ DEFINE LABEL MAPPING (from given table)
# -------------------------------------------------------------
label_mapping = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs (ascending)",
    5: "stairs (descending)",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling (sit)",
    14: "cycling (stand)",
    130: "cycling (sit, inactive)",
    140: "cycling (stand, inactive)"
}

with open(LABEL_MAP_PATH, "wb") as f:
    pickle.dump(label_mapping, f)

# -------------------------------------------------------------
# 4️⃣ DEFINE MODELS
# -------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42)
}

# -------------------------------------------------------------
# 5️⃣ TRAIN, EVALUATE, SAVE
# -------------------------------------------------------------
results = []
print("\nTraining and evaluating models...\n")

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Train Time (s)": round(train_time, 3)
    })

    # Save model
    model_filename = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    print(f"{name} ✅ | Accuracy: {acc:.4f} | Time: {train_time:.2f}s")

# -------------------------------------------------------------
# 6️⃣ SAVE RESULTS
# -------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
results_df.to_csv(RESULTS_PATH, index=False)

print("\nAll models trained and saved successfully!")
print(f"Results saved to: {RESULTS_PATH}")
print("\nSummary:\n", results_df)
