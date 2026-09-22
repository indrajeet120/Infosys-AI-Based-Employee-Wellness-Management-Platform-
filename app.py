"""
Unified Streamlit Web Application:
Infosys AI-Based Employee Wellness & Emotion Management Platform.
Combines all three milestones in a single, interactive, production-grade interface:
- Milestone 1: Multi-Channel Text Ingestion, NLP Preprocessing & VADER Sentiment Baseline
- Milestone 2: BERT / DistilBERT 6-Emotion Multi-Label Deep Learning Classification & ISEAR Evaluation
- Milestone 3: Emotion Intensity, Polarity, SentenceTransformer Semantic Matching & Personalized Wellness Recommendations
"""

from io import StringIO
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

from services.config import (
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    BERT_METRICS_PATH,
    DISTILBERT_METRICS_PATH,
    MODEL_COMPARISON_PATH,
    ISEAR_METRICS_PATH,
    MEDICAL_DISCLAIMER,
    DEFAULT_RANKING_WEIGHTS,
)
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment
from services.emotion import EmotionClassifier, get_emotion_classifier
from services.intensity import analyze_emotional_state, EmotionalState
from services.hybrid_recommender import get_recommendation_engine, get_personalized_recommendations
from services.user_profile import get_user_profile_manager, check_collaborative_filtering_availability
from services.reporting import process_pipeline_items
from services.trend_analysis import (
    calculate_emotion_frequency,
    calculate_intensity_trends,
    determine_dominant_emotions,
    calculate_polarity_trends,
    detect_repeated_patterns,
    build_tracked_user_state,
)
from services.feedback_learning import get_feedback_manager

# Page configuration
st.set_page_config(
    page_title="AI Employee Wellness & Emotion Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.0rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }
    .milestone-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
    }
    .rec-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        border-left: 5px solid #10B981;
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
    .badge-score {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
    }
    .badge-strat {
        background-color: #E5E7EB;
        color: #374151;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
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
    st.markdown('<div class="main-header">🧠 AI-Based Employee Wellness & Emotion Management Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All-in-One Dashboard: Milestone 1 (Sentiment) + Milestone 2 (Deep Emotions) + Milestone 3 (Personalized Wellness System)</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)

    # Initialize User Profile Manager & Feedback Learning Manager
    user_mgr = get_user_profile_manager()
    fb_mgr = get_feedback_manager()

    # Sidebar: Global Navigation & Settings
    st.sidebar.header("⚙️ Platform Navigation")
    app_section = st.sidebar.radio(
        "Select Dashboard View:",
        [
            "🌟 All-in-One Live Analysis & Recommendations",
            "📊 Batch Dataset & CSV Analysis",
            "📈 User Emotion History & Trend Analytics",
            "🏆 Model Evaluation & ISEAR Benchmarks",
            "👤 User Profile & Preference Manager",
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Model Configuration")
    user_id = st.sidebar.text_input("Active User / Employee ID:", value="employee_01")
    
    emotion_model_choice = st.sidebar.selectbox(
        "Transformer Emotion Model:",
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
    )

    top_k = st.sidebar.slider("Number of Recommendations (Top-K):", min_value=1, max_value=8, value=4)

    # Custom Ranking Weights Expander in Sidebar (Task 7 Configurable Feedback Weights)
    with st.sidebar.expander("⚖️ Customize Hybrid Ranking & Feedback Weights"):
        w_emotion = st.slider("Emotion Relevance Weight", 0.0, 0.6, 0.25, 0.05)
        w_intensity = st.slider("Intensity Fit Weight", 0.0, 0.4, 0.15, 0.05)
        w_pref = st.slider("User Preference Weight", 0.0, 0.5, 0.15, 0.05)
        w_sim = st.slider("Semantic Similarity Weight", 0.0, 0.5, 0.15, 0.05)
        w_accept = st.slider("Historical Acceptance Weight (Task 7)", 0.0, 0.4, 0.10, 0.05)
        w_rating = st.slider("User Rating Weight (Task 7)", 0.0, 0.3, 0.05, 0.05)
        w_rej_pen = st.slider("Rejection Penalty Weight (Task 7)", 0.0, 0.5, 0.25, 0.05)

    custom_weights = {
        "emotion_weight": w_emotion,
        "intensity_weight": w_intensity,
        "preference_weight": w_pref,
        "similarity_weight": w_sim,
        "history_weight": 0.10,
        "acceptance_weight": w_accept,
        "rating_weight": w_rating,
        "rejection_penalty_weight": w_rej_pen,
        "novelty_weight": 0.05,
        "duplicate_penalty": 0.20,
        "low_relevance_penalty": 0.30,
    }

    # =========================================================================
    # VIEW 1: All-in-One Live Analysis & Recommendations (Single Unified View)
    # =========================================================================
    if app_section == "🌟 All-in-One Live Analysis & Recommendations":
        st.subheader("💬 Live Employee State Analysis & Personalized Wellness Plan")

        # Preset Scenarios
        preset = st.selectbox(
            "Quick Scenario Presets (Optional):",
            [
                "Custom Text Input",
                "Work Overwhelm & Panic: 'I have multiple back-to-back project deadlines today and my heart is racing with panic!'",
                "Grief & Deep Exhaustion: 'I feel completely heartbroken, drained, and struggling to find motivation.'",
                "Frustration & Anger: 'My proposal was rejected without any explanation and I am furious!'",
                "Career Milestone Joy: 'I got selected for my dream project leadership role and I am thrilled!'",
                "Mixed Excitement & Anxiety: 'I am excited about starting my new role but terrified of moving to a new city.'",
            ]
        )

        if preset == "Custom Text Input":
            default_text = "I am feeling excited about the new project opportunities but slightly nervous about meeting the tight deadlines."
        else:
            default_text = preset.split(": '")[1].rstrip("'")

        user_text = st.text_area(
            "Enter text or employee feedback:",
            value=default_text,
            height=100,
            placeholder="Type your current thoughts, feelings, or workplace feedback..."
        )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("🚀 Analyze All Milestones & Recommend", use_container_width=True)

        if run_btn or user_text:
            with st.spinner("Executing complete NLP pipeline (Milestones 1, 2, & 3)..."):
                # 1. Milestone 1 Processing: Preprocessing & VADER
                preprocessed_text = preprocess_text(user_text)
                vader_res = analyze_sentiment(preprocessed_text)

                # 2. Milestone 2 & 3 Processing: Emotion Intensity, State & Recommendations
                rec_result = get_personalized_recommendations(
                    text=user_text,
                    user_id=user_id,
                    model_type=model_type_key,
                    top_k=top_k,
                    weights=custom_weights,
                )

            if not rec_result.get("is_valid", False):
                st.error(rec_result.get("error_message", "Invalid input"))
                return

            state = rec_result["emotional_state"]

            st.markdown("---")
            st.markdown("### 📊 Multi-Milestone Integrated Intelligence")

            # 3 Columns for Milestone 1, Milestone 2, and Milestone 3 State
            col_m1, col_m2, col_m3 = st.columns(3)

            # --- Milestone 1 Column ---
            with col_m1:
                st.markdown("""
                <div class="milestone-card">
                    <h4 style="color:#1E3A8A; margin-top:0;">💬 Milestone 1: VADER Sentiment</h4>
                    <p style="color:#6B7280; font-size:0.85rem;">Rule-based polarity with negation preservation</p>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Classification:** {render_sentiment_badge(vader_res['sentiment'])}", unsafe_allow_html=True)
                st.write("")
                vc1, vc2 = st.columns(2)
                vc1.metric("Compound", f"{vader_res['compound']:+.4f}")
                vc2.metric("Positive", f"{vader_res['pos']:.3f}")
                vc3, vc4 = st.columns(2)
                vc3.metric("Negative", f"{vader_res['neg']:.3f}")
                vc4.metric("Neutral", f"{vader_res['neu']:.3f}")
                
                st.caption(f"**Processed Text (Negations Preserved):** `{preprocessed_text}`")
                st.markdown("</div>", unsafe_allow_html=True)

            # --- Milestone 2 Column ---
            with col_m2:
                st.markdown(f"""
                <div class="milestone-card">
                    <h4 style="color:#1E3A8A; margin-top:0;">🤖 Milestone 2: Transformer Emotions</h4>
                    <p style="color:#6B7280; font-size:0.85rem;">Multi-label classification ({model_type_key.upper()})</p>
                """, unsafe_allow_html=True)
                
                primary_emo = state.get("dominant_emotion", "N/A")
                primary_conf = state.get("dominant_confidence", 0.0)
                st.markdown(f"**Primary Emotion:** {render_emotion_badge(primary_emo)} *(Conf: {primary_conf*100:.1f}%)*", unsafe_allow_html=True)
                
                detected_list = [f"{e['emotion']} ({e['confidence']*100:.0f}%)" for e in state.get("top_emotions", []) if e['confidence'] >= emotion_threshold]
                st.caption(f"**Detected Emotions (≥{emotion_threshold:.2f}):** {', '.join(detected_list) if detected_list else 'None'}")
                
                st.write("**Probability Distribution:**")
                for emo in EMOTIONS:
                    disp = EMOTION_DISPLAY_NAMES[emo]
                    p = float(state["probabilities"].get(disp, 0.0))
                    c_lbl, c_bar = st.columns([1, 2])
                    c_lbl.caption(f"**{disp}:** {p*100:.1f}%")
                    c_bar.progress(min(max(p, 0.0), 1.0))

                st.markdown("</div>", unsafe_allow_html=True)

            # --- Milestone 3 Column ---
            with col_m3:
                st.markdown("""
                <div class="milestone-card">
                    <h4 style="color:#065F46; margin-top:0;">🌿 Milestone 3: Intensity & State</h4>
                    <p style="color:#6B7280; font-size:0.85rem;">Dynamic intensity, polarity & severity mapping</p>
                """, unsafe_allow_html=True)

                st.metric("Emotional Intensity", f"{state['emotional_intensity']:.3f}", f"Severity: {state['emotion_severity']}")
                
                ic1, ic2 = st.columns(2)
                ic1.metric("Pos Polarity", f"{state['positive_polarity']:.2f}")
                ic2.metric("Neg Polarity", f"{state['negative_polarity']:.2f}")

                if state["mixed_emotion"]:
                    st.info(f"🔄 **Mixed State:** {state['final_emotional_state']}")
                else:
                    st.success(f"📌 **State:** {state['final_emotional_state']}")

                st.markdown("</div>", unsafe_allow_html=True)

            # --- Task 6: Historical Emotional Trend & Pattern Tracking Insights ---
            tracked_state = rec_result.get("tracked_user_state", {})
            if tracked_state:
                st.markdown("""
                <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1.0rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1E293B;">📈 Task 6: Historical State & Pattern Tracking</h4>
                        <span style="background-color: #EEF2FF; color: #4338CA; padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 600; font-size: 0.8rem;">
                            Historical Dominant: {dom_emo} | Confidence: {conf_pct:.1f}%
                        </span>
                    </div>
                    <div style="margin-top: 0.6rem; font-size: 0.9rem; color: #334155;">
                        <strong>Polarity Trajectory:</strong> Positive Trend: <code>{pos_dir}</code> | Negative Trend: <code>{neg_dir}</code> | Total Check-ins: <code>{total_ci}</code>
                    </div>
                </div>
                """.format(
                    dom_emo=tracked_state.get("dominant_emotion", "Joy"),
                    conf_pct=tracked_state.get("confidence", 0.80) * 100,
                    pos_dir=tracked_state.get("positive_trend", {}).get("direction", "stable").capitalize(),
                    neg_dir=tracked_state.get("negative_trend", {}).get("direction", "stable").capitalize(),
                    total_ci=tracked_state.get("total_historical_checkins", 1),
                ), unsafe_allow_html=True)

                patterns = tracked_state.get("repeated_patterns", [])
                if patterns:
                    st.write("**Detected Emotional Patterns & Streaks:**")
                    for p in patterns:
                        p_color = "#DC2626" if p.get("severity") == "High" else "#D97706"
                        st.markdown(
                            f"- <span style='color:{p_color}; font-weight:600;'>[{p.get('severity', 'Moderate')} Severity] {p.get('name', 'Pattern')}</span>: {p.get('description', '')}",
                            unsafe_allow_html=True,
                        )

            # --- Milestone 3 Personalized Recommendations ---
            st.markdown("---")
            st.markdown("### 🌟 Milestone 3: Tailored Wellness Activities & Interactive Feedback")
            
            recs = rec_result.get("recommendations", [])
            if not recs:
                st.warning("No activities met the minimum relevance cutoff for this state.")
            else:
                for idx, rec in enumerate(recs, start=1):
                    rec_title = rec.get("recommendation", rec["title"])
                    rec_score = rec.get("score", rec["final_score"])
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #111827;">#{idx} {rec_title}</h4>
                            <span class="badge-score">Match Score: {rec_score:.3f}</span>
                        </div>
                        <p style="color: #4B5563; margin: 0.4rem 0 0.6rem 0;">{rec['description']}</p>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Activity:</strong> <code>{rec['activity_type']}</code> | 
                            <strong>Duration:</strong> <code>{rec['duration']}</code> | 
                            <strong>Difficulty:</strong> <code>{rec['difficulty']}</code>
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Source Strategies:</strong> {' '.join([f'<span class="badge-strat">{s}</span>' for s in rec['source_strategies']])}
                        </div>
                        <div style="background-color: #F3F4F6; padding: 0.6rem 0.9rem; border-radius: 6px; border-left: 4px solid #4F46E5; margin-bottom: 0.5rem;">
                            💡 <strong>Why this was recommended:</strong><br/>
                            <em style="color: #1F2937;">{rec['reason']}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Task 8 Expandable Explanation Factors Breakdown
                    exp_factors = rec.get("explanation_factors", [])
                    if exp_factors:
                        with st.expander(f"📊 Detailed Explanation Factors for #{idx} {rec_title}"):
                            icon_map = {
                                "emotion": "🎭",
                                "intensity": "⚡",
                                "preference": "⭐",
                                "historical_behavior": "📜",
                                "content_relevance": "🔍",
                                "previous_feedback": "🔄",
                            }
                            for factor in exp_factors:
                                f_type = factor.get("factor_type", "")
                                f_icon = icon_map.get(f_type, "📌")
                                f_name = factor.get("name", "Factor")
                                f_score = factor.get("score", 0.0)
                                f_contrib = factor.get("weighted_contribution", 0.0)
                                f_detail = factor.get("detail", "")
                                f_strong = factor.get("is_strong", False)

                                badge_str = "🔥 **[Strong Contributor]** " if f_strong else ""
                                st.markdown(
                                    f"{f_icon} **{f_name}** ({badge_str}Score: `{f_score:.2f}` | Contrib: `{f_contrib:.3f}`)\n"
                                    f"> _{f_detail}_\n"
                                )

                    # Task 7 Interactive Feedback & Rating Controls
                    user_accepted_ids = fb_mgr.get_accepted_content_ids(user_id)
                    user_rejected_ids = fb_mgr.get_rejected_content_ids(user_id)
                    item_id = rec['content_id']

                    status_str = "⚪ New"
                    if item_id in user_accepted_ids:
                        status_str = "✅ Previously Accepted"
                    elif item_id in user_rejected_ids:
                        status_str = "❌ Previously Rejected"

                    st.markdown(f"**Feedback Status:** `{status_str}` | **Historical Acceptance Score:** `{rec.get('historical_acceptance', 0.50):.2f}`")

                    fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([1.2, 1.2, 2.0, 1.2])

                    with fb_col1:
                        if st.button(f"✅ Accept", key=f"accept_btn_{item_id}_{idx}", use_container_width=True):
                            fb_mgr.record_acceptance(
                                user_id=user_id,
                                recommendation_id=item_id,
                                recommendation_type=rec['activity_type'],
                                emotion=state.get('dominant_emotion', 'Joy'),
                                intensity=state.get('emotional_intensity', 0.5),
                                rating=5.0,
                                metadata={"title": rec['title'], "tags": rec.get('tags', [])},
                            )
                            st.success(f"Accepted '{rec['title']}'! Recommendation relevance boosted.")

                    with fb_col2:
                        if st.button(f"❌ Reject", key=f"reject_btn_{item_id}_{idx}", use_container_width=True):
                            fb_mgr.record_rejection(
                                user_id=user_id,
                                recommendation_id=item_id,
                                recommendation_type=rec['activity_type'],
                                emotion=state.get('dominant_emotion', 'Joy'),
                                intensity=state.get('emotional_intensity', 0.5),
                                reason="not_relevant",
                                metadata={"title": rec['title'], "tags": rec.get('tags', [])},
                            )
                            st.warning(f"Rejected '{rec['title']}'. This activity will be suppressed.")

                    with fb_col3:
                        star_rating = st.selectbox(
                            "Rating:",
                            [5, 4, 3, 2, 1],
                            format_func=lambda x: f"{'⭐' * x} ({x} Star{'s' if x > 1 else ''})",
                            key=f"star_sel_{item_id}_{idx}",
                            label_visibility="collapsed",
                        )

                    with fb_col4:
                        if st.button(f"⭐ Submit", key=f"rate_btn_{item_id}_{idx}", use_container_width=True):
                            fb_mgr.record_rating(
                                user_id=user_id,
                                recommendation_id=item_id,
                                rating=float(star_rating),
                                recommendation_type=rec['activity_type'],
                                emotion=state.get('dominant_emotion', 'Joy'),
                                intensity=state.get('emotional_intensity', 0.5),
                                metadata={"title": rec['title'], "tags": rec.get('tags', [])},
                            )
                            st.info(f"Recorded {star_rating}-Star rating for '{rec['title']}'!")

    # =========================================================================
    # VIEW 2: Batch Dataset & CSV Analysis
    # =========================================================================
    elif app_section == "📊 Batch Dataset & CSV Analysis":
        st.subheader("📊 Batch Processing & Tabular Pipeline Report")

        input_type = st.radio(
            "Select Input Source:",
            ["Upload CSV File", "Upload TXT File", "Load Benchmark Sample Corpus (9 samples)"],
            horizontal=True,
        )

        ingested_items = []
        if input_type == "Upload CSV File":
            st.info("CSV file must contain a column named **`text`**.")
            csv_file = st.file_uploader("Upload CSV:", type=["csv"])
            if csv_file:
                ingested_items = ingest_csv_file(csv_file, text_column="text", filename=csv_file.name)

        elif input_type == "Upload TXT File":
            txt_file = st.file_uploader("Upload TXT (one sentence per line):", type=["txt"])
            if txt_file:
                ingested_items = ingest_txt_file(txt_file, filename=txt_file.name)

        elif input_type == "Load Benchmark Sample Corpus (9 samples)":
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

            # Summary Metrics
            st.subheader("📈 Summary Statistics")
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Total", stats["total_samples"])
            m2.metric("Valid", stats["valid_samples"])
            m3.metric("Invalid", stats["invalid_samples"])
            m4.metric("Analyzed", stats["analyzed_samples"])
            m5.metric("Positive 😊", stats["positive_count"])
            m6.metric("Negative 😞", stats["negative_count"])
            m7.metric("Neutral 😐", stats["neutral_count"])

            st.markdown("### 📋 Combined Tabular Report")
            st.dataframe(report_df, use_container_width=True)

            csv_buf = StringIO()
            report_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Full Report as CSV",
                data=csv_buf.getvalue(),
                file_name="sentiment_emotion_wellness_report.csv",
                mime="text/csv",
            )

    # =========================================================================
    # VIEW 3: Emotion History & Trend Analytics (Task 6)
    # =========================================================================
    elif app_section == "📈 User Emotion History & Trend Analytics":
        st.subheader(f"📈 Emotion History, Trends & Pattern Analytics for User: `{user_id}`")
        prof = user_mgr.get_or_create_profile(user_id)

        if not prof.emotion_history:
            st.info("No historical emotion check-ins recorded yet. Use the '🌟 All-in-One Live Analysis' tab to log your thoughts and generate history.")
        else:
            history_records = prof.emotion_history
            hist_df = pd.DataFrame(history_records)

            # Window configuration
            win_col, _ = st.columns([2, 4])
            with win_col:
                window_size = st.slider(
                    "Recent Moving Window (Check-ins):",
                    min_value=2,
                    max_value=max(3, len(history_records)),
                    value=min(5, max(2, len(history_records))),
                )

            # Compute Task 6 Analytics
            freq_data = calculate_emotion_frequency(history_records, window_size=None)
            recent_freq_data = calculate_emotion_frequency(history_records, window_size=window_size)
            intensity_data = calculate_intensity_trends(history_records, window_size=window_size)
            dominant_data = determine_dominant_emotions(history_records, window_size=None)
            polarity_data = calculate_polarity_trends(history_records, recent_window=window_size)
            patterns = detect_repeated_patterns(history_records)

            # 1. Top Key Summary Metrics
            st.markdown("#### 📌 Key Emotional Tracking Indicators")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Check-ins", len(history_records))
            k2.metric("Dominant Emotion", dominant_data["dominant_emotion"], f"Score: {dominant_data['dominant_score']:.2f}")
            k3.metric("Baseline Intensity", f"{intensity_data['historical_mean']:.3f}")
            k4.metric(
                f"Recent Moving Avg (N={window_size})",
                f"{intensity_data['recent_mean']:.3f}",
                f"{intensity_data['delta']:+.3f} ({intensity_data['direction']})",
            )
            k5.metric(
                "Polarity Status",
                polarity_data["net_polarity_status"],
                f"Shift: {polarity_data['polarity_shift']:+.2f}",
            )

            st.markdown("---")

            # 2. Emotion Frequency Distribution
            st.markdown("#### 📊 1. Historical Emotion Frequency (6 Core Categories)")
            f_col1, f_col2 = st.columns([3, 2])
            
            with f_col1:
                freq_chart_df = pd.DataFrame({
                    "Emotion": list(freq_data["counts"].keys()),
                    "All-Time Count": list(freq_data["counts"].values()),
                    f"Recent (Last {window_size})": [recent_freq_data["counts"].get(e, 0) for e in freq_data["counts"].keys()],
                }).set_index("Emotion")
                st.bar_chart(freq_chart_df)

            with f_col2:
                st.write("**Frequency Breakdown:**")
                for item in freq_data["ranked_emotions"]:
                    emo = item["emotion"]
                    cnt = item["count"]
                    pct = item["proportion"] * 100
                    st.write(f"- **{emo}:** {cnt} times ({pct:.1f}%)")

            st.markdown("---")

            # 3. Emotion Intensity & Polarity Trajectory Over Time
            st.markdown("#### 📈 2. Emotion Intensity & Polarity Progression Over Time")
            t_col1, t_col2 = st.columns(2)

            with t_col1:
                st.write("**Intensity Progression:**")
                if "intensity" in hist_df.columns:
                    plot_df = pd.DataFrame({
                        "Intensity": hist_df["intensity"].astype(float),
                        "Historical Baseline": [intensity_data["historical_mean"]] * len(hist_df),
                    })
                    st.line_chart(plot_df)
                    st.caption(f"Intensity Volatility (StdDev): `{intensity_data['volatility']:.3f}` | Trajectory: `{intensity_data['direction'].upper()}`")

            with t_col2:
                st.write("**Positive vs Negative Polarity:**")
                if "positive_polarity" in hist_df.columns and "negative_polarity" in hist_df.columns:
                    pol_df = pd.DataFrame({
                        "Positive Polarity": hist_df["positive_polarity"].astype(float),
                        "Negative Polarity": hist_df["negative_polarity"].astype(float),
                    })
                    st.line_chart(pol_df)
                    st.caption(f"Positive Trend: `{polarity_data['positive_trend']['direction']}` | Negative Trend: `{polarity_data['negative_trend']['direction']}`")

            st.markdown("---")

            # 4. Detected Repeated Patterns & Streaks
            st.markdown("#### 🔁 3. Detected Emotional Patterns & Behavioral Streaks")
            if not patterns:
                st.success("✅ No adverse chronic patterns or persistent distress streaks detected. State is stable.")
            else:
                for p in patterns:
                    p_color = "#FEF2F2" if p.get("severity") == "High" else "#FFFBEB"
                    b_color = "#EF4444" if p.get("severity") == "High" else "#F59E0B"
                    t_color = "#991B1B" if p.get("severity") == "High" else "#92400E"
                    st.markdown(f"""
                    <div style="background-color: {p_color}; border-left: 5px solid {b_color}; padding: 0.8rem 1.0rem; border-radius: 6px; margin-bottom: 0.6rem;">
                        <h4 style="margin: 0; color: {t_color};">[{p.get('severity', 'Moderate')} Severity] {p.get('name', 'Pattern')}</h4>
                        <p style="margin: 0.3rem 0; color: #374151;">{p.get('description', '')}</p>
                        <small style="color: #6B7280;"><strong>Recommendation Action:</strong> <code>{p.get('recommendation_hint', 'adaptive_support')}</code></small>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # 5. Detailed Historical Records Table
            st.markdown("#### 📋 4. Detailed Historical Logs")
            st.dataframe(hist_df, use_container_width=True)

            csv_hist_buf = StringIO()
            hist_df.to_csv(csv_hist_buf, index=False)
            st.download_button(
                label="📥 Download Emotion History as CSV",
                data=csv_hist_buf.getvalue(),
                file_name=f"emotion_history_{user_id}.csv",
                mime="text/csv",
            )

    # =========================================================================
    # VIEW 4: Model Evaluation & ISEAR Benchmarks
    # =========================================================================
    elif app_section == "🏆 Model Evaluation & ISEAR Benchmarks":
        st.subheader("🏆 Model Evaluation & Benchmark Results")
        tab_comp, tab_isear, tab_collab = st.tabs([
            "⚖️ BERT vs DistilBERT Comparison",
            "🌍 Held-Out ISEAR Benchmark Evaluation",
            "🤝 Collaborative Filtering Status",
        ])

        with tab_comp:
            if MODEL_COMPARISON_PATH.exists():
                with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                st.success(f"🏆 **Winner:** **{comp_data.get('better_performing_model', 'N/A')}** (Evaluated by {comp_data.get('comparison_metric', 'Macro F1')})")
                metrics_df = pd.DataFrame(comp_data.get("metrics_summary", {})).T
                st.dataframe(metrics_df, use_container_width=True)
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
                    st.markdown("#### 📋 ISEAR Per-Sample Predictions Table")
                    isear_df = pd.read_csv("reports/isear_results.csv")
                    st.dataframe(isear_df, use_container_width=True)
            else:
                st.info("Run `python scripts/validate_isear.py` to generate ISEAR validation reports.")

        with tab_collab:
            collab_info = check_collaborative_filtering_availability()
            if collab_info["available"]:
                st.success(f"✅ Collaborative Filtering Active ({collab_info['total_users']} users, {collab_info['total_interactions']} ratings)")
            else:
                st.warning(f"⚠️ **Collaborative Filtering Status:** Unavailable (Documented Fallback Active)")
                st.write(f"**Reason:** {collab_info['reason']}")
                st.write(f"**Active Fallback:** `{collab_info['fallback_strategy']}`")

    # =========================================================================
    # VIEW 5: User Profile & Preference Manager
    # =========================================================================
    elif app_section == "👤 User Profile & Preference Manager":
        st.subheader(f"⚙️ Profile & Preference Settings: `{user_id}`")
        curr_prof = user_mgr.get_or_create_profile(user_id)

        all_content_types = ["exercise", "audio", "text", "interactive_guide"]
        all_activities = [
            "breathing exercise", "meditation", "journaling", "relaxation activity",
            "motivational content", "productivity break", "physical activity",
            "positive reflection", "music/audio activity"
        ]

        pref_types = st.multiselect(
            "Preferred Content Types:",
            options=all_content_types,
            default=[t for t in curr_prof.preferred_content_types if t in all_content_types],
        )

        pref_acts = st.multiselect(
            "Preferred Activity Types:",
            options=all_activities,
            default=[a for a in curr_prof.preferred_activities if a in all_activities],
        )

        pref_lang = st.selectbox("Preferred Language:", ["English", "Spanish", "French", "German", "Hindi"], index=0)

        if st.button("💾 Save User Preferences"):
            user_mgr.update_preferences(
                user_id=user_id,
                preferred_content_types=pref_types,
                preferred_activities=pref_acts,
                preferred_language=pref_lang,
            )
            fb_mgr.record_preference_change(
                user_id=user_id,
                preferred_content_types=pref_types,
                preferred_activities=pref_acts,
                preferred_language=pref_lang,
            )
            st.success("User preferences updated successfully and logged in feedback history!")

        st.markdown("---")
        st.markdown("#### 📋 Task 7: Recommendation Interaction & Feedback Learning History")
        
        user_fb_events = fb_mgr.get_user_feedback(user_id)
        if not user_fb_events:
            st.info("No recommendation feedback recorded yet. Accept, reject, or rate recommendations in the '🌟 All-in-One Live Analysis' tab.")
        else:
            fb_list = [e.to_dict() for e in user_fb_events]
            fb_df = pd.DataFrame(fb_list)

            accepted_cnt = sum(1 for e in user_fb_events if e.accepted)
            rejected_cnt = sum(1 for e in user_fb_events if e.rejected)
            rated_events = [e.rating for e in user_fb_events if e.rating is not None]
            avg_rating_str = f"{np.mean(rated_events):.2f} ⭐" if rated_events else "N/A"

            cf1, cf2, cf3, cf4 = st.columns(4)
            cf1.metric("Total Events Logged", len(user_fb_events))
            cf2.metric("Accepted Recommendations ✅", accepted_cnt)
            cf3.metric("Rejected Recommendations ❌", rejected_cnt)
            cf4.metric("Average Rating ⭐", avg_rating_str)

            st.markdown("##### 📜 Detailed Feedback Event Log")
            st.dataframe(fb_df, use_container_width=True)

            csv_fb_buf = StringIO()
            fb_df.to_csv(csv_fb_buf, index=False)
            st.download_button(
                label="📥 Download Recommendation Feedback History as CSV",
                data=csv_fb_buf.getvalue(),
                file_name=f"feedback_history_{user_id}.csv",
                mime="text/csv",
            )

        # --- Task 9: ML Validation & Benchmark Comparison Section ---
        st.markdown("---")
        st.markdown("### 🧪 Task 9: Advanced ML Validation & Performance Testing")
        st.markdown("Offline comparative evaluation of **Baseline Rule Recommender** vs **Advanced Hybrid ML Engine** using 15 controlled test profiles and explicit ground-truth annotations.")

        report_json_path = REPORTS_DIR / "recommendation_evaluation_report.json"
        
        col_btn1, col_btn2 = st.columns([2, 3])
        with col_btn1:
            if st.button("🚀 Run Offline Recommendation Benchmark (Task 9)", use_container_width=True):
                with st.spinner("Running evaluation benchmark across 15 test profiles..."):
                    from services.recommendation_eval import run_recommendation_evaluation
                    run_recommendation_evaluation(k=3)
                    st.success("Benchmark completed! Report saved to `reports/recommendation_evaluation_report.json`.")

        if report_json_path.exists():
            with open(report_json_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)

            cfg = rep_data.get("evaluation_configuration", {})
            b_m = rep_data.get("baseline_metrics", {})
            a_m = rep_data.get("advanced_metrics", {})
            imp = rep_data.get("improvement_pct", {})
            k_val = cfg.get("k", 3)

            st.markdown(f"**Benchmark Status:** `✅ Completed` | **K:** `{k_val}` | **Test Profiles:** `{cfg.get('num_test_users', 15)}` | **Candidate Pool:** `{cfg.get('candidate_pool_size', 24)} items`")

            # Comparative Metrics Table
            metrics_table_data = [
                {"Metric": f"Precision@{k_val}", "Baseline": f"{b_m.get('precision_at_k', 0):.4f}", "Advanced (Hybrid ML)": f"{a_m.get('precision_at_k', 0):.4f}", "Improvement": f"+{imp.get('precision_at_k', 0):.2f}%"},
                {"Metric": f"Recall@{k_val}", "Baseline": f"{b_m.get('recall_at_k', 0):.4f}", "Advanced (Hybrid ML)": f"{a_m.get('recall_at_k', 0):.4f}", "Improvement": f"+{imp.get('recall_at_k', 0):.2f}%"},
                {"Metric": f"F1-Score@{k_val}", "Baseline": f"{b_m.get('f1_score_at_k', 0):.4f}", "Advanced (Hybrid ML)": f"{a_m.get('f1_score_at_k', 0):.4f}", "Improvement": f"+{imp.get('f1_score_at_k', 0):.2f}%"},
                {"Metric": f"NDCG@{k_val} (Ranking Quality)", "Baseline": f"{b_m.get('ndcg_at_k', 0):.4f}", "Advanced (Hybrid ML)": f"{a_m.get('ndcg_at_k', 0):.4f}", "Improvement": f"+{imp.get('ndcg_at_k', 0):.2f}%"},
                {"Metric": "Ground-Truth Acceptance Proxy", "Baseline": f"{b_m.get('ground_truth_acceptance_proxy', 0):.4f}", "Advanced (Hybrid ML)": f"{a_m.get('ground_truth_acceptance_proxy', 0):.4f}", "Improvement": f"+{imp.get('ground_truth_acceptance_proxy', 0):.2f}%"},
                {"Metric": "Recommendation Diversity", "Baseline": f"{b_m.get('diversity_score', 0):.4f}", "Advanced (Hybrid ML)": f"{a_m.get('diversity_score', 0):.4f}", "Improvement": f"{imp.get('diversity_score', 0):.2f}%"},
                {"Metric": "Mean Latency (ms)", "Baseline": f"{b_m.get('mean_latency_ms', 0):.2f} ms", "Advanced (Hybrid ML)": f"{a_m.get('mean_latency_ms', 0):.2f} ms", "Improvement": f"+{imp.get('mean_latency_diff_ms', 0):.2f} ms overhead"},
                {"Metric": "P95 Latency (ms)", "Baseline": f"{b_m.get('p95_latency_ms', 0):.2f} ms", "Advanced (Hybrid ML)": f"{a_m.get('p95_latency_ms', 0):.2f} ms", "Improvement": f"+{imp.get('p95_latency_diff_ms', 0):.2f} ms overhead"},
            ]
            st.dataframe(pd.DataFrame(metrics_table_data), use_container_width=True)

            with st.expander("🔍 View Active ML Signals & Report Files"):
                st.write("**Active Advanced Model Signals:**")
                for sig in cfg.get("active_advanced_signals", []):
                    st.write(f"- `{sig}`")
                st.info(f"📁 Reports saved at:\n- JSON: `reports/recommendation_evaluation_report.json`\n- Markdown: `reports/recommendation_evaluation_report.md`")
        else:
            st.info("Click '🚀 Run Offline Recommendation Benchmark (Task 9)' to compute and view live ML validation metrics.")


if __name__ == "__main__":
    main()
