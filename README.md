# 🌿 MoodMentor: AI-Based Employee Wellness & Emotion Management Platform

> **Milestone 4 Final Delivery & Comprehensive System Documentation**

---

> [!WARNING]
> ### ⚠️ Medical & Clinical Disclaimer
> **MoodMentor is an AI-driven text analysis and wellness recommendation tool intended solely for workplace emotional self-awareness, stress management, and non-clinical personal support. It is NOT a medical diagnostic tool, psychiatric assessment system, or clinical intervention platform. It does not provide medical diagnoses, treatment plans, or crisis intervention services.**

---

## 1. High-Level System Vision

**MoodMentor** is an end-to-end, enterprise-grade AI Employee Wellness & Emotion Management Platform. It transforms multi-channel textual inputs (survey responses, daily check-ins, team feedback, incident reports) into structured emotional intelligence insights and delivers dynamic, highly personalized wellness micro-interventions (e.g., box breathing, reflective journaling, physical walks, audio relaxation).

The platform progresses systematically across four distinct architectural milestones:
- **Milestone 1:** Data Ingestion, Text Cleaning & Rule-Based VADER Sentiment Baseline
- **Milestone 2:** Fine-Tuned Transformer Multi-Label Emotion Classification (BERT & DistilBERT across 6 Core Ekman Emotions)
- **Milestone 3:** Advanced Emotion Intensity, SentenceTransformer Semantic Search & Hybrid Re-Ranking Recommendation Engine
- **Milestone 4:** Packaging, Automated Testing, Interactive Streamlit Dashboard, Feedback Learning, Stress Testing & Security Validation

---

## 2. Medical & Clinical Disclaimer
*(See prominent notice above)*  
MoodMentor estimates emotional states solely based on linguistic patterns in user-submitted text. It includes hardcoded safety guardrails that detect severe crisis keywords and immediately direct users to professional mental health resources without attempting clinical advice.

---

## 3. Core Architectural Principles

1. **Modular Service Architecture:** Clean separation of concerns across Data Ingestion, NLP Preprocessing, Sentiment Analysis, Multi-Label Emotion Classification, Intensity Estimation, Hybrid Recommendation, Feedback Learning, Reporting, and Security.
2. **100% CPU Compatibility & Optional GPU Acceleration:** Runs cleanly on Standard CPU environments with automatic CUDA detection when available.
3. **Preserved Storage & API Contracts:** All JSON file schemas (`user_profiles.json`, `feedback_events.json`, `wellness_activities.json`) maintain backward compatibility without breaking existing data structures.
4. **Empirical Ground-Truth Metric Calculation:** All reported precision, recall, F1, latency, and throughput metrics are dynamically calculated from actual test execution without hardcoded values.
5. **Privacy-First Scoping & Data Sanitization:** Strict user-level access controls, HTML/XSS input sanitization, and automatic secret/token redaction.

---

## 4. Key Capabilities & Technical Highlights

- **Multi-Channel Text Ingestion:** Raw strings, uploaded `.txt` logs, and tabular `.csv` batch datasets.
- **Negation-Preserving NLP Pipeline:** Preprocessing preserves negations ("not happy", "never calm") critical for accurate sentiment and emotion scoring.
- **Fine-Tuned Dual-Transformer Engine:** Comparative evaluation between `bert-base-uncased` and `distilbert-base-uncased` fine-tuned on multi-label emotion data.
- **SentenceTransformer Semantic Retrieval:** `all-MiniLM-L6-v2` dense embeddings for cosine similarity matching against wellness activities.
- **8-Factor Hybrid Re-Ranking Formula:** Combines emotion match score, intensity delta fit, user preferences, semantic similarity, historical acceptance, explicit star ratings, rejection penalties, and novelty boosts.
- **Interactive Feedback & Continuous Learning:** Closed-loop acceptance, rejection, and rating tracking that updates recommendation scores dynamically.
- **Export & Reporting Suite:** Automated tabular CSV summaries and ReportLab PDF executive reports.

---

## 5. Repository Structure

```
.
├── app.py                         # Unified Streamlit Web Application (6 Interactive Views)
├── moodmentor/                    # Packaging Module & Command-Line Interface
│   ├── __init__.py
│   └── cli.py                     # CLI Entry Point (moodmentor run/evaluate/verify)
├── services/                      # Modular Business Logic & ML Engines
│   ├── config.py                  # Paths, Emotion Constants & Weight Defaults
│   ├── dataset_loader.py          # Data Loading & Label Binarization Utilities
│   ├── emotion.py                  # Transformer Emotion Classifier Wrapper (PyTorch)
│   ├── feedback_learning.py       # Feedback Event Tracker & Continuous Learning Engine
│   ├── filtering.py               # Advanced Search & Multi-Criteria Filter Engine
│   ├── hybrid_recommender.py      # Hybrid Recommendation & Re-Ranking Engine
│   ├── ingestion.py                # Multi-Channel Data Ingestion Handlers
│   ├── intensity.py               # Emotion Intensity, Polarity & Severity Estimator
│   ├── model_stress_testing.py    # Latency, Throughput & Stress Testing Benchmark
│   ├── preprocessing.py           # NLP Text Cleaning & Negation Handling
│   ├── recommendation_eval.py     # Ground-Truth ML Recommendation Benchmark (K=3)
│   ├── reporting.py               # CSV Data Processing & ReportLab PDF Export Engine
│   ├── security.py                # Input Sanitization, User Scoping & Secret Redaction
│   ├── semantic.py                # SentenceTransformer Embedding & Semantic Search
│   ├── sentiment.py               # VADER Rule-Based Sentiment Analysis Baseline
│   ├── trend_analysis.py          # Historical Trajectory, Period Aggregation & Streaks
│   ├── user_profile.py            # User Profile Manager & Collaborative Filtering Fallback
│   └── wellness_data.py           # Wellness Content Repository & Metadata Index
├── tests/                         # PyTest Unit & Integration Test Suite (140+ Tests)
│   ├── test_emotion.py
│   ├── test_end_to_end_integration.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   ├── test_feedback_learning.py
│   ├── test_filtering.py
│   ├── test_hybrid_recommender.py
│   ├── test_ingestion.py
│   ├── test_integration_milestone2.py
│   ├── test_integration_milestone3.py
│   ├── test_intensity.py
│   ├── test_isear.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_recommendation_evaluation.py
│   ├── test_security.py
│   ├── test_semantic.py
│   ├── test_sentiment.py
│   ├── test_stress_performance.py
│   ├── test_trend_analysis.py
│   └── test_user_profile.py
├── scripts/                       # Training, Validation & Evaluation Scripts
│   ├── evaluate_models.py
│   ├── validate_isear.py
│   └── evaluate_recommendations.py
├── verify_milestone1.py           # Verification Script for Milestone 1
├── verify_milestone2.py           # Verification Script for Milestone 2
├── verify_milestone3.py           # Verification Script for Milestone 3
├── pyproject.toml                 # Modern Package Build Configuration
├── setup.py                       # Development Installation Setup
└── README.md                      # Complete Project Documentation
```

---

## 6. Environment Requirements & Prerequisites

- **OS:** Windows 10/11, Linux, or macOS
- **Python Version:** Python 3.10+ (Tested on Python 3.11.9)
- **Key Dependencies:** `streamlit`, `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `nltk`, `reportlab`, `pytest`

---

## 7. Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/indrajeet120/Infosys-AI-Based-Employee-Wellness-Management-Platform-.git
   cd Infosys-AI-Based-Employee-Wellness-Management-Platform-
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Package in Development Mode:**
   ```bash
   python setup.py develop
   ```

---

## 8. Configuration Management

Configuration settings reside in `services/config.py`:
- **Emotions Supported:** Joy, Sadness, Anger, Fear, Surprise, Disgust
- **Default Cutoff Confidence:** `0.50`
- **Default Top-K Recommendations:** `4`
- **Model Paths:** Fine-tuned checkpoints saved under `models/` directory.

---

## 9. Data Pipeline & Multi-Channel Input Processing (Milestone 1)

The pipeline ingests single strings, text files (`.txt`), and structured tabular CSV files containing employee feedback. Input items are wrapped into a normalized `IngestedItem` structure with metadata tracking character count, line numbers, and validity status.

---

## 10. NLP Preprocessing & VADER Sentiment Baseline (Milestone 1)

Text cleaning removes noise and URLs while explicitly preserving contractive negations ("not", "never", "n't") to prevent false positive flips. VADER sentiment analysis calculates `pos`, `neg`, `neu`, and `compound` polarity scores to classify text into Positive, Negative, or Neutral.

---

## 11. Multi-Label Deep Learning Classification Engine (Milestone 2)

Fine-tuned `BERT` and `DistilBERT` models output independent sigmoid probabilities for all 6 core emotions simultaneously, supporting complex mixed-emotion detection (e.g., high Joy + high Fear during a career promotion).

---

## 12. Held-Out ISEAR Evaluation & Benchmark Performance (Milestone 2)

Evaluated on the held-out ISEAR benchmark corpus:
- **Sample Accuracy:** 70.0%
- **Hamming Accuracy:** 0.8800
- **Macro F1-Score:** 0.7250

---

## 13. Emotion Intensity & Severity Estimation Engine (Milestone 3)

Calculates dynamic emotional intensity on a normalized scale `[0.0, 1.0]` combining transformer confidence, sentiment magnitude, and expressive punctuation indicators. Categorizes severity into **Low**, **Moderate**, and **High**.

---

## 14. SentenceTransformer Semantic Search & Content Indexing (Milestone 3)

Uses `all-MiniLM-L6-v2` dense embeddings to perform semantic vector search, computing cosine similarity between employee emotional state descriptions and wellness activity titles, descriptions, and tags.

---

## 15. Hybrid Recommendation Engine Architecture & Re-Ranking Formula (Milestone 3)

Calculates a comprehensive hybrid match score using 8 distinct factors:
$$\text{FinalScore} = w_{\text{emo}} \cdot S_{\text{emo}} + w_{\text{int}} \cdot S_{\text{int}} + w_{\text{pref}} \cdot S_{\text{pref}} + w_{\text{sim}} \cdot S_{\text{sim}} + w_{\text{acc}} \cdot S_{\text{acc}} + w_{\text{rat}} \cdot S_{\text{rat}} - w_{\text{rej}} \cdot P_{\text{rej}} + \text{NoveltyBoost}$$

---

## 16. Dynamic Cold-Start, Collaborative Filtering & Rule Fallbacks (Milestone 3)

For new users without historical ratings, the engine applies dynamic cold-start rules based on explicit preferences and emotion-category mapping. When collaborative user-item matrix data is insufficient, it seamlessly transitions to content-based semantic matching.

---

## 17. Interactive Feedback & Continuous Learning Framework (Milestone 4 Task 3)

Users can Accept, Reject, or Star-rate recommendations directly in the Streamlit UI. Feedback events update historical acceptance scores (`S_acc`) and apply rejection penalties (`P_rej`) to adaptively suppress unwanted activity types over time.

---

## 18. Advanced Search, Filtering & Query Capabilities (Milestone 4 Task 4)

`FilterEngine` enables multi-criteria filtering across stored check-in history, recommendation logs, feedback events, and wellness activities by text keyword, date range, emotion category, difficulty level, and intensity bounds.

---

## 19. Tabular & PDF Analytics Report Generation (Milestone 4 Task 5)

Generates clean CSV data tables and executive PDF reports via ReportLab, embedding summary metrics, dominant emotion distributions, and activity recommendations.

---

## 20. Unified Streamlit Web Dashboard User Guide (Milestone 4 Task 1)

Launch dashboard via CLI or Streamlit:
```bash
python -m moodmentor.cli run
```
**Dashboard Sections:**
1. 🌟 **All-in-One Live Analysis & Recommendations**
2. 📊 **Batch Dataset & CSV Analysis**
3. 📈 **User Emotion History & Trend Analytics**
4. 🏆 **Model Evaluation & ISEAR Benchmarks**
5. ⚡ **Stress Testing & Security Diagnostics**
6. 👤 **User Profile & Preference Manager**

---

## 21. CLI Usage & Automated Verification Protocols (Milestone 4 Task 9)

```bash
# Display help and version
python -m moodmentor.cli --help

# Run interactive dashboard
python -m moodmentor.cli run

# Execute offline recommendation benchmark
python -m moodmentor.cli evaluate

# Run all milestone verification scripts
python -m moodmentor.cli verify
```

---

## 22. System Stress Testing, Latency & Scaling Benchmarks (Milestone 4 Task 7)

Evaluated via `ModelPerformanceStressTester` on CPU:
- **Small Workload (10 requests):** Throughput ~8.5 QPS | Avg Latency ~118 ms
- **Medium Workload (50 requests):** Throughput ~8.2 QPS | Avg Latency ~121 ms
- **Large Workload (200 requests):** Throughput ~8.0 QPS | Avg Latency ~124 ms | Memory Delta < 1.0 MB

---

## 23. Security, Input Sanitization & Privacy Compliance (Milestone 4 Task 8)

- **Input Sanitization:** Strips HTML, `<script>` tags, and clamps maximum text length to prevent XSS and buffer attacks.
- **User Scoping:** Enforces user isolation so employees can only access their own emotion history and feedback records.
- **Secret Redaction:** Automatically redacts API keys, tokens, and passwords from logs using pattern matching.

---

## 24. Verification Evidence & Summary Metrics Report (Milestone 4 Task 10)

- **Total Automated Unit & Integration Tests:** 140+ passing tests (100% pass rate).
- **Offline ML Recommendation Evaluation (K=3):**
  - **Baseline Rule Recommender Precision@3:** 0.4444
  - **Advanced Hybrid ML Engine Precision@3:** 0.8889 (+100.0% Improvement)
  - **NDCG@3 Score:** 0.9245 (+85.2% Improvement)
- **All Milestone Verification Scripts Passed:** `verify_milestone1.py`, `verify_milestone2.py`, `verify_milestone3.py`.
