"""
Streamlit Application for Milestone 2:
Multi-Label Deep Emotion Classification (BERT & DistilBERT), Dynamic Probabilities,
Model Comparative Evaluation, and ISEAR Benchmark Validation.
"""

from io import StringIO
from pathlib import Path
import json
import streamlit as st
import pandas as pd

from services.config import EMOTIONS, EMOTION_DISPLAY_NAMES, BERT_METRICS_PATH, DISTILBERT_METRICS_PATH, MODEL_COMPARISON_PATH, ISEAR_METRICS_PATH
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment
from services.emotion import EmotionClassifier, get_emotion_classifier
from services.reporting import process_pipeline_items

# Page configuration
st.set_page_config(
    page_title="Deep Emotion Classifier | Milestone 2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-pos {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-neg {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-neu {
        background-color: #E5E7EB;
        color: #374151;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-emo {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_emotion_classifier(model_type: str) -> EmotionClassifier:
    """Loads and caches the Transformer Emotion Classifier."""
    return get_emotion_classifier(model_type)


def render_sentiment_badge(sentiment: str) -> str:
    if sentiment == "Positive":
        return '<span class="badge-pos">😊 Positive</span>'
    elif sentiment == "Negative":
        return '<span class="badge-neg">😞 Negative</span>'
    elif sentiment == "Neutral":
        return '<span class="badge-neu">😐 Neutral</span>'
    else:
        return f'<span>{sentiment}</span>'


def render_emotion_badge(emotion: str) -> str:
    emoji_map = {
        "Joy": "😄 Joy",
        "Sadness": "😢 Sadness",
        "Anger": "😡 Anger",
        "Fear": "😨 Fear",
        "Surprise": "😲 Surprise",
        "Disgust": "🤢 Disgust",
    }
    label = emoji_map.get(emotion, emotion)
    return f'<span class="badge-emo">{label}</span>'


def main():
    st.markdown('<div class="main-header">🧠 Deep Emotion Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Milestone 2 — Multi-Label Transformer Emotion Classification (BERT / DistilBERT), Dynamic Probabilities & Benchmark Validation</div>', unsafe_allow_html=True)

    # Sidebar Navigation & Settings
    st.sidebar.header("⚙️ Milestone 2 Settings")
    app_section = st.sidebar.radio(
        "Select View:",
        [
            "⚡ Live Emotion Classifier (Inference)",
            "📊 BERT vs DistilBERT Model Comparison",
            "🏆 Held-Out ISEAR Benchmark Evaluation",
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Configuration")
    emotion_model_choice = st.sidebar.selectbox(
        "Select Transformer Architecture:",
        ["DistilBERT (distilbert-base-uncased)", "BERT (bert-base-uncased)"],
        index=0,
    )
    model_type_key = "distilbert" if "DistilBERT" in emotion_model_choice else "bert"

    emotion_threshold = st.sidebar.slider(
        "Multi-Label Confidence Cutoff:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Emotions with probability >= cutoff will be included in Detected Emotions.",
    )

    # Section 1: Model Comparison
    if app_section == "📊 BERT vs DistilBERT Model Comparison":
        st.subheader("📊 BERT vs DistilBERT Multi-Label Evaluation")
        if MODEL_COMPARISON_PATH.exists():
            with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
            
            st.success(f"🏆 **Winner:** **{comp_data.get('better_performing_model', 'N/A')}** (Evaluated by {comp_data.get('comparison_metric', 'Macro F1')})")
            
            metrics_dict = comp_data.get("metrics_summary", {})
            metrics_df = pd.DataFrame(metrics_dict).T
            st.dataframe(metrics_df, use_container_width=True)
            
            col_b, col_d = st.columns(2)
            with col_b:
                st.markdown("#### 📘 BERT Emotion-Wise Metrics")
                st.json(comp_data.get("bert_details", {}).get("emotion_metrics", {}))
            with col_d:
                st.markdown("#### 📙 DistilBERT Emotion-Wise Metrics")
                st.json(comp_data.get("distilbert_details", {}).get("emotion_metrics", {}))
        else:
            st.warning("Comparison report not found. Run `python scripts/evaluate_models.py` to generate it.")
        return

    # Section 2: ISEAR Benchmark
    if app_section == "🏆 Held-Out ISEAR Benchmark Evaluation":
        st.subheader("🏆 Held-Out ISEAR Benchmark Validation")
        if ISEAR_METRICS_PATH.exists():
            with open(ISEAR_METRICS_PATH, "r", encoding="utf-8") as f:
                isear_metrics = json.load(f)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Benchmark Samples", isear_metrics.get("total_samples", 0))
            m2.metric("Sample Accuracy", f"{isear_metrics.get('sample_accuracy', 0) * 100:.1f}%")
            m3.metric("Hamming Accuracy", f"{isear_metrics.get('hamming_accuracy', 0):.4f}")
            m4.metric("Macro F1-Score", f"{isear_metrics.get('macro_f1', 0):.4f}")
            
            if Path("reports/isear_results.csv").exists():
                st.markdown("#### 📋 ISEAR Per-Sample Predictions Table")
                isear_df = pd.read_csv("reports/isear_results.csv")
                st.dataframe(isear_df, use_container_width=True)
        else:
            st.warning("ISEAR report not found. Run `python scripts/validate_isear.py` to generate it.")
        return

    # Section 3: Live Inference
    input_source = st.selectbox(
        "Choose Input Format:",
        ["Direct Text Input", "Upload TXT File", "Upload CSV Dataset", "Load Standard Benchmark Dataset"],
    )

    ingested_items = []
    if input_source == "Direct Text Input":
        default_example = "I am thrilled about my new promotion but slightly scared of the immense responsibility."
        user_text = st.text_area(
            "Enter text for deep multi-label emotion analysis:",
            value=default_example,
            height=110,
        )
        if st.button("🚀 Analyze Emotions", use_container_width=True) or user_text:
            ingested_items = ingest_text(user_text, source="direct")

    elif input_source == "Upload TXT File":
        uploaded_file = st.file_uploader("Upload .txt file:", type=["txt"])
        if uploaded_file:
            ingested_items = ingest_txt_file(uploaded_file, filename=uploaded_file.name)

    elif input_source == "Upload CSV Dataset":
        uploaded_file = st.file_uploader("Upload .csv file with 'text' column:", type=["csv"])
        if uploaded_file:
            ingested_items = ingest_csv_file(uploaded_file, text_column="text", filename=uploaded_file.name)

    elif input_source == "Load Standard Benchmark Dataset":
        sample_path = "data/sample_corpus.csv"
        ingested_items = ingest_csv_file(sample_path, text_column="text")

    if ingested_items:
        st.markdown("---")
        load_cached_emotion_classifier(model_type_key)

        report_df, stats = process_pipeline_items(
            ingested_items,
            include_emotion=True,
            emotion_model_type=model_type_key,
            emotion_threshold=emotion_threshold,
        )

        # Single Record View
        if len(report_df) == 1 and report_df.iloc[0]["Status"] == "Valid":
            row = report_df.iloc[0]
            st.markdown("### 🔍 Emotion Prediction Inspector")
            
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("**Input Text:**")
                st.info(row["Input Text"])
                st.markdown(f"**VADER Baseline Sentiment:** {render_sentiment_badge(row['Sentiment'])} &nbsp; *(Compound: {row['Compound']:.4f})*", unsafe_allow_html=True)
                st.markdown(f"**Final Combined Output:**\n`{row.get('Combined Analysis', '')}`")

            with c_right:
                st.markdown(f"**Active Model:** `{row.get('Emotion Model', model_type_key.upper())}`")
                primary_emo = row.get("Primary Emotion", "N/A")
                primary_conf = row.get("Primary Confidence", 0.0)
                st.markdown(f"**Primary Emotion:** {render_emotion_badge(primary_emo)} &nbsp; *(Confidence: {primary_conf * 100:.1f}%)*", unsafe_allow_html=True)
                st.markdown(f"**Detected Emotions ($\ge {emotion_threshold:.2f}$):** `{row.get('Detected Emotions', 'None')}`")

                st.markdown("**6-Emotion Probability Distribution:**")
                for emo in EMOTIONS:
                    disp = EMOTION_DISPLAY_NAMES[emo]
                    prob = row.get(f"Prob_{disp}", 0.0)
                    prob_val = float(prob) if prob is not None else 0.0
                    col_e1, col_e2 = st.columns([1, 3])
                    with col_e1:
                        st.write(f"**{disp}:** {prob_val * 100:.1f}%")
                    with col_e2:
                        st.progress(min(max(prob_val, 0.0), 1.0))

        # Full Table View
        st.markdown("### 📋 Multi-Label Emotion Report Table")
        display_cols = [
            "ID", "Input Text", "Sentiment", "Primary Emotion",
            "Primary Confidence", "Detected Emotions"
        ]
        for emo in EMOTIONS:
            p_col = f"Prob_{EMOTION_DISPLAY_NAMES[emo]}"
            if p_col in report_df.columns:
                display_cols.append(p_col)

        st.dataframe(report_df[[c for c in display_cols if c in report_df.columns]], use_container_width=True)

        csv_buffer = StringIO()
        report_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Milestone 2 Report as CSV",
            data=csv_buffer.getvalue(),
            file_name="milestone2_emotion_report.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
