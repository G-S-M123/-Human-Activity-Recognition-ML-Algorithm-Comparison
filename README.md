## 🧠 Human Activity Recognition — ML Algorithm Comparison

### 📘 Overview

This project compares multiple **machine learning algorithms** on the **HARTH (Human Activity Recognition)** dataset, which contains wearable sensor data from thigh and back IMUs.

The aim is to analyze model performance across **Accuracy, Precision, Recall, F1-Score, and Training Time**, visualize results, and test model predictions interactively using a **Streamlit dashboard**.

---

### 🎯 Objectives

* Compare the performance of classical supervised ML algorithms.
* Visualize results interactively on a Streamlit UI.
* Provide a simple way to test predictions using real HARTH samples.
* Export a summarized comparison report for viva / submission.

---

### 🧩 Algorithms Compared

| Algorithm                    | Type           | Notes                       |
| ---------------------------- | -------------- | --------------------------- |
| Logistic Regression          | Linear         | Baseline classifier         |
| K-Nearest Neighbors (KNN)    | Non-parametric | Distance-based              |
| Naive Bayes                  | Probabilistic  | Simple, fast                |
| Decision Tree                | Non-linear     | Interpretable               |
| Random Forest                | Ensemble       | Robust and stable           |
| Support Vector Machine (SVM) | Kernel-based   | Strong with non-linear data |

---

### 📂 Folder Structure

```
ML_Comparison/
│
├── harth70/
│   ├── .csv                 # HARTH dataset (or multiple CSVs)
│   └── ...
│
├── models/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── scaler.pkl
│   └── ...
│
├── results/
│   └── comparison_results.csv    # Metrics summary for all models
│
├── train_and_compare.py          # Core training + evaluation script
├── app.py                        # Streamlit dashboard
└── README.md                     # Project documentation
```

---

### ⚙️ Setup Instructions

#### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, use:

```bash
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib
```

#### 2️⃣ Prepare data

Place the HARTH dataset file(s) under:

```
./data/harth.csv
```

Each file should have columns similar to:

```
timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
```

---

#### 3️⃣ Train and Compare Models

Run the training script to generate results:

```bash
python train_and_compare.py
```

This will:

* Train all models
* Save `.pkl` models in `./models`
* Generate `comparison_results.csv` in `./results`

---

#### 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run app.py
```

The app provides:

* 📊 Model performance visualizations (Plotly + Seaborn)
* 📈 Combined metric comparison charts
* 🧪 “Try Sample Data” section for live testing
* 📄 Downloadable performance report

---

### 🧠 Label Mapping

| Label | Activity                  | Notes                      |
| :---- | :------------------------ | :------------------------- |
| 1     | Walking                   | —                          |
| 2     | Running                   | —                          |
| 3     | Shuffling                 | Standing with leg movement |
| 4     | Stairs (Ascending)        | —                          |
| 5     | Stairs (Descending)       | —                          |
| 6     | Standing                  | —                          |
| 7     | Sitting                   | —                          |
| 8     | Lying                     | —                          |
| 13    | Cycling (Sit)             | —                          |
| 14    | Cycling (Stand)           | —                          |
| 130   | Cycling (Sit, Inactive)   | Without leg movement       |
| 140   | Cycling (Stand, Inactive) | Without leg movement       |

---

### 📊 Key Visualizations

* **Accuracy Comparison** — Interactive bar chart (Plotly)
* **Radar Chart** — Compare Accuracy, Precision, Recall, F1-Score
* **Heatmap** — Visual correlation of model metrics
* **Combined Metrics Chart** — Line + bar visualization
* **Training vs Prediction Time Plot**

---

### 🧩 Sample Testing UI

In the Streamlit app, you can:

* Select a real sample from the dataset
* Choose a trained model
* Predict activity + view true label and confidence
* Optionally view class probability distribution

Example output:

```
✅ Predicted Activity: Standing
🎯 True Label: Standing
🔹 Confidence: 94.2%
```

---

### 📈 Example Result Summary (from comparison_results.csv)

| Model         | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
| :------------ | :------- | :-------- | :----- | :------- | :------------- |
| Random Forest | 0.94     | 0.94      | 0.94   | 0.94     | 3.21           |
| SVM           | 0.91     | 0.91      | 0.90   | 0.90     | 5.40           |
| KNN           | 0.88     | 0.88      | 0.87   | 0.87     | 0.15           |
| ...           | ...      | ...       | ...    | ...      | ...            |

---

### 🧰 Future Work

* Include all 15 HARTH files for larger-scale evaluation.
* Add **deep learning models (LSTM, CNN)** for comparison.
* Use **cross-validation** for more robust accuracy metrics.
* Optimize model efficiency for **edge deployment on UAV or wearable devices**.

---