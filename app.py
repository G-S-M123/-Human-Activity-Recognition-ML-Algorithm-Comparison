import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import seaborn as sns
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
MODEL_DIR = "./models"
RESULTS_PATH = "./results/comparison_results.csv"

st.set_page_config(page_title="HARTH ML Comparison", layout="wide")
st.title("🤖 HARTH Human Activity Recognition — ML Algorithm Comparison Dashboard")

# -------------------------------------------------------------
# LOAD RESULTS
# -------------------------------------------------------------
if not os.path.exists(RESULTS_PATH):
    st.error("Results not found. Please run train_and_compare.py first.")
    st.stop()

results_df = pd.read_csv(RESULTS_PATH)
label_map = pickle.load(open(os.path.join(MODEL_DIR, "label_mapping.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# -------------------------------------------------------------
# PERFORMANCE SUMMARY
# -------------------------------------------------------------
st.subheader("📋 Model Performance Summary")
st.dataframe(results_df.style.background_gradient(cmap="Blues"), use_container_width=True)

# -------------------------------------------------------------
# ADVANCED VISUALIZATIONS
# -------------------------------------------------------------
st.subheader("📊  Visualizations")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Accuracy Chart",
    "🧮 Radar Chart",
    "🔥 Metrics Heatmap",
    "📊 All Metrics Overview"
])

with tab1:
    st.markdown("### Accuracy Comparison (Interactive)")
    fig_bar = px.bar(
        results_df,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison",
        text="Accuracy",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown("### Radar Comparison (Accuracy, Precision, Recall, F1)")
    radar_df = results_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                               var_name="Metric", value_name="Score")
    fig_radar = px.line_polar(
        radar_df,
        r="Score",
        theta="Metric",
        color="Model",
        line_close=True,
        template="plotly_dark",
        markers=True
    )
    fig_radar.update_traces(fill="toself", opacity=0.6)
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.markdown("### Heatmap of Model Metrics")
    heatmap_data = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

with tab4:
    st.markdown("### Combined Metrics Visualization (Bar + Line Mix)")
    
    # Normalize time columns for better visual scaling (since they're in seconds)
    df_viz = results_df.copy()
    df_viz["Training Time (s)"] = df_viz["Train Time (s)"].astype(float)
    # df_viz["Prediction Time (s)"] = df_viz["Prediction Time (s)"].astype(float)

    fig_combo = px.bar(
        df_viz,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Metric Comparison (Accuracy, Precision, Recall, F1)",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_combo.update_layout(
        yaxis_title="Score",
        legend_title="Metric",
        height=500
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    st.markdown("### ⏱ Training vs Prediction Time (Line Chart)")
    fig_time = px.line(
        df_viz,
        x="Model",
        y=["Training Time (s)"],
        markers=True,
        title="Training vs Prediction Time per Model",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_time.update_layout(yaxis_title="Seconds", legend_title="Time Metric")
    st.plotly_chart(fig_time, use_container_width=True)

# -------------------------------------------------------------
# SAMPLE TEST PREDICTION
# -------------------------------------------------------------

st.markdown("---")
st.header("🧪 Try Out Sample Data")

# Load a few test samples directly from dataset
data_path = r"F:\10_study\Harth_Algo_compare\har70plus\501.csv"  
if os.path.exists(data_path):
    df_sample = pd.read_csv(data_path).dropna()
    
    # pick columns dynamically
    feature_cols = [c for c in df_sample.columns if c not in ["timestamp", "label"]]
    
    # prepare a small random subset (10 samples)
    sample_subset = df_sample.sample(10, random_state=100).reset_index(drop=True)
    
    st.markdown("Select a real sample from the dataset to test model predictions:")
    sample_index = st.selectbox("Choose a sample:", range(len(sample_subset)))
    
    chosen_sample = sample_subset.iloc[sample_index]
    
    st.write("### 📈 Sensor Readings (features)")
    st.dataframe(chosen_sample.to_frame().T, use_container_width=True)

    # load models list
    model_files = [f for f in os.listdir("./models") if f.endswith(".pkl") and f != "scaler.pkl"]
    model_choice = st.selectbox("Choose a model to test:", model_files)

    # Prediction logic
    if st.button("🔍 Predict Activity"):
        # load model + scaler
        with open(os.path.join("./models", model_choice), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join("./models", "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        
        # preprocess and predict
        X_sample = scaler.transform([chosen_sample[feature_cols].values])
        pred = model.predict(X_sample)[0]
        
        # label mapping
        label_map = {
            1: "Walking", 2: "Running", 3: "Shuffling",
            4: "Stairs (Ascending)", 5: "Stairs (Descending)",
            6: "Standing", 7: "Sitting", 8: "Lying",
            13: "Cycling (Sit)", 14: "Cycling (Stand)",
            130: "Cycling (Sit, Inactive)", 140: "Cycling (Stand, Inactive)"
        }
        true_label = chosen_sample["label"]
        pred_label = label_map.get(pred, f"Unknown ({pred})")
        true_label_text = label_map.get(true_label, f"Unknown ({true_label})")

        # confidence (if supported)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sample)[0]
            confidence = np.max(probs)
        else:
            confidence = None

        st.success(f"**Predicted Activity:** {pred_label}")
        st.success(f"**Predicted Activity label:** {pred}")
        if confidence:
            st.write(f"**Confidence:** {confidence:.2%}")
        st.write(f"**True Label:** {true_label_text}")
else:
    st.warning("Dataset not found. Please ensure './data/harth.csv' exists.")

# -------------------------------------------------------------
# PDF REPORT
# -------------------------------------------------------------
st.subheader("📄 Generate Model Comparison Report")

def generate_pdf(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>HARTH Dataset Model Comparison Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "This report presents a comparison of various supervised machine learning algorithms applied to "
        "the HARTH dataset for human activity recognition. It includes key metrics such as Accuracy, "
        "Precision, Recall, and F1-Score.",
        styles["Normal"])
    )
    elements.append(Spacer(1, 12))

    table_data = [df.columns.tolist()] + df.values.tolist()
    elements.append(Table(table_data))
    elements.append(Spacer(1, 12))

    best_model = df.iloc[df["Accuracy"].idxmax()]["Model"]
    best_acc = df["Accuracy"].max()
    elements.append(Paragraph(f"<b>Best Model:</b> {best_model} (Accuracy: {best_acc:.4f})", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

if st.button("Generate PDF Report"):
    pdf = generate_pdf(results_df)
    st.download_button(
        "⬇️ Download HARTH Comparison Report",
        pdf,
        file_name="HARTH_Model_Comparison_Report.pdf",
        mime="application/pdf"
    )
