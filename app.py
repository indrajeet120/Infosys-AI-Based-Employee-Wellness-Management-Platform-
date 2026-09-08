"""
Streamlit Application for Sentiment Analysis & Deep Emotion Classification.
Integrates Milestone 1 (VADER Baseline Sentiment) and Milestone 2 (BERT / DistilBERT Multi-label Emotion Classification).
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
    page_title="Sentiment & Deep Emotion Analyzer",
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
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
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
    st.markdown('<div class="main-header">💬 Sentiment & Deep Emotion Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Milestones 1 & 2 — VADER Baseline Sentiment + Multi-Label Transformer Emotion Classification (BERT / DistilBERT)</div>', unsafe_allow_html=True)

    # Sidebar Navigation & Settings
    st.sidebar.header("⚙️ Pipeline Configuration")
    input_mode = st.sidebar.radio(
        "Select Input Method:",
        [
            "Direct Text Input",
            "Upload TXT File",
            "Upload CSV File",
            "Load Benchmark Sample Corpus",
            "📊 View Model Evaluations & Benchmark Reports",
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Deep Emotion Settings")
    emotion_model_choice = st.sidebar.selectbox(
        "Select Transformer Model:",
        ["BERT (bert-base-uncased)", "DistilBERT (distilbert-base-uncased)"],
        index=0,
    )
    model_type_key = "distilbert" if "DistilBERT" in emotion_model_choice else "bert"

    emotion_threshold = st.sidebar.slider(
        "Multi-Label Confidence Threshold:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Emotions with probability >= threshold will be flagged as detected.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Classification Rules")
    st.sidebar.markdown("""
    - **Sentiment (VADER):**
      - `Positive`: Compound $\ge 0.05$
      - `Negative`: Compound $\le -0.05$
      - `Neutral`: Otherwise
    - **Emotions (Multi-Label):**
      - Joy, Sadness, Anger, Fear, Surprise, Disgust
      - Independent Sigmoid Probabilities
      - Primary Emotion: Max Probability
    """)

    # 1. View Model Evaluations & Reports Page
    if input_mode == "📊 View Model Evaluations & Benchmark Reports":
        st.subheader("📊 Model Evaluation & Benchmark Reports")
        
        tab_comp, tab_isear = st.tabs(["⚖️ BERT vs DistilBERT Comparison", "🏆 ISEAR Benchmark Results"])
        
        with tab_comp:
            if MODEL_COMPARISON_PATH.exists():
                with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                
                st.success(f"🏆 **Better Performing Model:** {comp_data.get('better_performing_model', 'N/A')} (based on {comp_data.get('comparison_metric', 'Macro F1')})")
                
                # Metrics Table
                metrics_dict = comp_data.get("metrics_summary", {})
                metrics_df = pd.DataFrame(metrics_dict).T
                st.dataframe(metrics_df, use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 📘 BERT Details")
                    st.json(comp_data.get("bert_details", {}).get("emotion_metrics", {}))
                with c2:
                    st.markdown("#### 📙 DistilBERT Details")
                    st.json(comp_data.get("distilbert_details", {}).get("emotion_metrics", {}))
            else:
                st.info("Run `python scripts/evaluate_models.py` to generate comparative evaluation reports.")

        with tab_isear:
            if ISEAR_METRICS_PATH.exists():
                with open(ISEAR_METRICS_PATH, "r", encoding="utf-8") as f:
                    isear_metrics = json.load(f)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Benchmark Samples", isear_metrics.get("total_samples", 0))
                m2.metric("Sample Accuracy", f"{isear_metrics.get('sample_accuracy', 0) * 100:.1f}%")
                m3.metric("Hamming Accuracy", f"{isear_metrics.get('hamming_accuracy', 0):.4f}")
                m4.metric("Macro F1-Score", f"{isear_metrics.get('macro_f1', 0):.4f}")
                
                if Path("reports/isear_results.csv").exists():
                    st.markdown("#### 📋 ISEAR Predictions Table")
                    isear_df = pd.read_csv("reports/isear_results.csv")
                    st.dataframe(isear_df, use_container_width=True)
            else:
                st.info("Run `python scripts/validate_isear.py` to generate ISEAR validation reports.")
        return

    # 2. Main Processing Section
    ingested_items = []

    if input_mode == "Direct Text Input":
        st.subheader("📝 Direct Text Input")
        default_example = "I am excited about the new job opportunity but slightly nervous about moving to a new city!"
        user_text = st.text_area(
            "Enter text for sentiment & multi-label emotion analysis:",
            value=default_example,
            height=110,
            placeholder="Type or paste your text here..."
        )
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            analyze_btn = st.button("🚀 Analyze Sentiment & Emotions", use_container_width=True)

        if analyze_btn or user_text:
            ingested_items = ingest_text(user_text, source="direct")

    elif input_mode == "Upload TXT File":
        st.subheader("📄 Upload TXT File")
        uploaded_file = st.file_uploader(
            "Upload a plain text (.txt) file:",
            type=["txt"],
            help="Upload a UTF-8 text file with one sentence/paragraph per line."
        )
        if uploaded_file is not None:
            filename = uploaded_file.name
            ingested_items = ingest_txt_file(uploaded_file, filename=filename, source="txt")

    elif input_mode == "Upload CSV File":
        st.subheader("📊 Upload CSV File")
        st.info("The CSV file must contain a column named **`text`**.")
        uploaded_file = st.file_uploader(
            "Upload a CSV (.csv) file:",
            type=["csv"],
            help="Upload a CSV file containing a 'text' column."
        )
        if uploaded_file is not None:
            filename = uploaded_file.name
            ingested_items = ingest_csv_file(uploaded_file, text_column="text", filename=filename, source="csv")

    elif input_mode == "Load Benchmark Sample Corpus":
        st.subheader("🧪 Benchmark Sample Corpus")
        st.write("Loads the standard 9-sample Milestone 1 & 2 benchmark corpus.")
        sample_path = "data/sample_corpus.csv"
        try:
            ingested_items = ingest_csv_file(sample_path, text_column="text", source="csv")
        except Exception as e:
            st.error(f"Error loading sample corpus: {e}")

    # Process and Render Results
    if ingested_items:
        st.markdown("---")
        # Ensure cached classifier is ready
        load_cached_emotion_classifier(model_type_key)

        report_df, stats = process_pipeline_items(
            ingested_items,
            include_emotion=True,
            emotion_model_type=model_type_key,
            emotion_threshold=emotion_threshold,
        )

        # Summary Metrics
        st.subheader("📈 Processing & Pipeline Summary")
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Total Samples", stats["total_samples"])
        m2.metric("Valid Samples", stats["valid_samples"])
        m3.metric("Invalid", stats["invalid_samples"])
        m4.metric("Analyzed", stats["analyzed_samples"])
        m5.metric("Positive 😊", stats["positive_count"])
        m6.metric("Negative 😞", stats["negative_count"])
        m7.metric("Neutral 😐", stats["neutral_count"])

        # Display Validation Errors if any
        invalid_rows = [item for item in ingested_items if not item.is_valid]
        if invalid_rows:
            st.warning(f"⚠️ Found {len(invalid_rows)} invalid input item(s) that were skipped:")
            for item in invalid_rows:
                st.error(f"• ID {item.id}: {item.error_message} (Input: '{item.text}')")

        # Single Record Detailed View
        if len(report_df) == 1 and report_df.iloc[0]["Status"] == "Valid":
            st.markdown("### 🔍 Comprehensive Analysis Inspector")
            row = report_df.iloc[0]

            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown("**Original Text:**")
                st.info(row["Input Text"])
                st.markdown("**Processed Text (Normalized for VADER):**")
                st.code(row["Processed Text"], language="text")

                st.markdown("#### 💬 VADER Sentiment Analysis")
                st.markdown(f"**Classification:** {render_sentiment_badge(row['Sentiment'])}", unsafe_allow_html=True)
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Positive", f"{row['Positive']:.3f}")
                sc2.metric("Negative", f"{row['Negative']:.3f}")
                sc3.metric("Neutral", f"{row['Neutral']:.3f}")
                sc4.metric("Compound", f"{row['Compound']:.3f}")

            with col_right:
                st.markdown(f"#### 🧠 Deep Emotion Analysis ({row.get('Emotion Model', model_type_key.upper())})")
                primary_emo = row.get("Primary Emotion", "N/A")
                primary_conf = row.get("Primary Confidence", 0.0)
                st.markdown(f"**Primary Emotion:** {render_emotion_badge(primary_emo)} &nbsp; *(Confidence: {primary_conf * 100:.1f}%)*", unsafe_allow_html=True)
                st.markdown(f"**Detected Emotions ($\ge {emotion_threshold:.2f}$):** `{row.get('Detected Emotions', 'None')}`")

                st.markdown("**Multi-Label Probability Distribution:**")
                for emo in EMOTIONS:
                    disp = EMOTION_DISPLAY_NAMES[emo]
                    prob = row.get(f"Prob_{disp}", 0.0)
                    prob_val = float(prob) if prob is not None else 0.0
                    col_e1, col_e2 = st.columns([1, 3])
                    with col_e1:
                        st.write(f"**{disp}:** {prob_val * 100:.1f}%")
                    with col_e2:
                        st.progress(min(max(prob_val, 0.0), 1.0))

            st.success(f"**Final Combined Summary:** {row.get('Combined Analysis', '')}")

        # Tabular Report
        st.markdown("### 📋 Combined Sentiment & Emotion Report")
        
        display_cols = [
            "ID", "Input Text", "Sentiment", "Compound",
            "Emotion Model", "Primary Emotion", "Primary Confidence", "Detected Emotions"
        ]
        # Add probability columns if present
        for emo in EMOTIONS:
            p_col = f"Prob_{EMOTION_DISPLAY_NAMES[emo]}"
            if p_col in report_df.columns:
                display_cols.append(p_col)

        display_df = report_df[[c for c in display_cols if c in report_df.columns]]

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Compound": st.column_config.NumberColumn(format="%.4f"),
                "Primary Confidence": st.column_config.NumberColumn(format="%.4f"),
            }
        )

        # Download Report CSV
        csv_buffer = StringIO()
        report_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Sentiment & Emotion Report as CSV",
            data=csv_buffer.getvalue(),
            file_name="sentiment_emotion_report.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
