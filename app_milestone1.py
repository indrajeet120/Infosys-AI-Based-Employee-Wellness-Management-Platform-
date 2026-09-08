"""
Streamlit Application for Milestone 1 ONLY:
Text Ingestion, NLP Preprocessing & Baseline Sentiment Analysis (VADER).
"""

from io import StringIO
import streamlit as st
import pandas as pd

from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment
from services.reporting import process_pipeline_items

# Page configuration
st.set_page_config(
    page_title="Text Sentiment Analyzer | Milestone 1",
    page_icon="💬",
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
</style>
""", unsafe_allow_html=True)


def render_sentiment_badge(sentiment: str) -> str:
    if sentiment == "Positive":
        return '<span class="badge-pos">😊 Positive</span>'
    elif sentiment == "Negative":
        return '<span class="badge-neg">😞 Negative</span>'
    elif sentiment == "Neutral":
        return '<span class="badge-neu">😐 Neutral</span>'
    else:
        return f'<span>{sentiment}</span>'


def main():
    st.markdown('<div class="main-header">💬 Text Sentiment Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Milestone 1 — Text Ingestion, NLP Preprocessing & Baseline VADER Sentiment</div>', unsafe_allow_html=True)

    # Sidebar Navigation & Settings
    st.sidebar.header("⚙️ Milestone 1 Configuration")
    input_mode = st.sidebar.radio(
        "Select Input Method:",
        ["Direct Text Input", "Upload TXT File", "Upload CSV File", "Load Benchmark Sample Corpus"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Classification Rules")
    st.sidebar.markdown("""
    - **Positive**: `Compound >= 0.05`
    - **Negative**: `Compound <= -0.05`
    - **Neutral**: `-0.05 < Compound < 0.05`
    """)

    # Main Processing Section
    ingested_items = []

    if input_mode == "Direct Text Input":
        st.subheader("📝 Direct Text Input")
        default_example = "I absolutely love this product! It exceeded all my expectations."
        user_text = st.text_area(
            "Enter text for sentiment analysis:",
            value=default_example,
            height=120,
            placeholder="Type or paste your text here..."
        )
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

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
        st.write("Loads the standard 9-sample Milestone 1 benchmark corpus.")
        sample_path = "data/sample_corpus.csv"
        try:
            ingested_items = ingest_csv_file(sample_path, text_column="text", source="csv")
        except Exception as e:
            st.error(f"Error loading sample corpus: {e}")

    # Process and Render Results
    if ingested_items:
        st.markdown("---")
        report_df, stats = process_pipeline_items(ingested_items, include_emotion=False)

        # Summary Metrics
        st.subheader("📈 Processing & Sentiment Summary")
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
            st.markdown("### 🔍 Detailed Analysis")
            row = report_df.iloc[0]

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Original Text:**")
                st.info(row["Input Text"])
                st.markdown("**Processed Text (Normalized & Lemmatized):**")
                st.code(row["Processed Text"], language="text")

            with c2:
                st.markdown(f"**Overall Classification:** {render_sentiment_badge(row['Sentiment'])}", unsafe_allow_html=True)
                st.write("")
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                score_col1.metric("Positive", f"{row['Positive']:.3f}")
                score_col2.metric("Negative", f"{row['Negative']:.3f}")
                score_col3.metric("Neutral", f"{row['Neutral']:.3f}")
                score_col4.metric("Compound", f"{row['Compound']:.3f}")

        # Tabular Report
        st.markdown("### 📋 Sentiment Classification Report")
        display_df = report_df[[
            "ID", "Input Text", "Processed Text", "Sentiment",
            "Positive", "Negative", "Neutral", "Compound", "Status"
        ]]

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Compound": st.column_config.NumberColumn(format="%.4f"),
                "Positive": st.column_config.NumberColumn(format="%.4f"),
                "Negative": st.column_config.NumberColumn(format="%.4f"),
                "Neutral": st.column_config.NumberColumn(format="%.4f"),
            }
        )

        # Download Report CSV
        csv_buffer = StringIO()
        display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv_buffer.getvalue(),
            file_name="milestone1_sentiment_report.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
