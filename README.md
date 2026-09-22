# AI Employee Wellness & Emotion Management Platform

An end-to-end NLP, Deep Learning, and Personalized Recommendation Platform integrating:
* **Milestone 1:** Multi-Channel Ingestion, Negation-Preserving Preprocessing & VADER Sentiment Baseline.
* **Milestone 2:** Fine-Tuned BERT & DistilBERT 6-Emotion Multi-Label Classification (`Joy`, `Sadness`, `Anger`, `Fear`, `Surprise`, `Disgust`) and ISEAR Benchmark Validation.
* **Milestone 3:** Advanced Emotion Intensity & State Analysis, SentenceTransformer Dense Semantic Matching, and Hybrid Personalized Wellness Recommendation Engine with Interactive Feedback Learning.

---

> ⚠️ **Medical Disclaimer:**  
> *This system estimates emotional states from text and recommends wellness content. It is not a medical diagnostic or clinical assessment system.*

---

## 🌟 Comprehensive Architecture Diagram

```text
                                [User Text / Review / Feedback]
                                                ↓
                            [Ingestion & Strict Input Validation]
                                                ↓
            ┌───────────────────────────────────┴───────────────────────────────────┐
            ↓                                                                       ↓
[NLP Preprocessing Layer]                                              [Preserved Raw Text]
• URL, HTML & Noise Cleaning                                           • WordPiece Tokenization
• Contraction Expansion                                                • Sequence Padding/Truncation
• Negation Preservation ("not happy" → "not happy")                                 ↓
• WordNet Lemmatization                                                [Fine-Tuned BERT / DistilBERT]
            ↓                                                          • Multi-Label Sigmoid Head
[VADER Baseline Sentiment]                                             • 6 Independent Probabilities
• Pos / Neg / Neu / Compound Polarity                                               ↓
            └───────────────────────────────────┬───────────────────────────────────┘
                                                ↓
                           [Emotion Intensity & State Engine]
                           • Dominant Emotion & Confidence
                           • Dynamic Intensity Score (0.0 to 1.0)
                           • Positive vs Negative Polarity Contrast
                           • Mixed Emotion Detection
                           • Severity Tiers (Low, Moderate, High, Very High)
                           • Final Descriptive Emotional State
                                                ↓
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       │                          HYBRID RECOMMENDATION ENGINE                           │
       │                                                                                 │
       │   1. Rule-Based Calibrator: Maps intensity to grounding vs reflective tools    │
       │   2. Content-Based Filter: Matches activity types, tags, and difficulties       │
       │   3. User Preference Matcher: Incorporates preferred content types & language   │
       │   4. Emotion Similarity: Computes overlap with detected emotion distribution    │
       │   5. Semantic Dense Matcher: SentenceTransformer (all-MiniLM-L6-v2) Cosine Sim │
       │   6. Historical Behavior: Boosts liked tags, penalizes disliked content         │
       │                                                                                 │
       │   Dynamic Multi-Factor Ranking Formula:                                         │
       │   Final Score = w_emo*S_emo + w_int*S_int + w_pref*S_pref + w_sim*S_sim         │
       │               + w_hist*S_hist + w_nov*S_nov - Dup_Penalty - LowRel_Penalty      │
       └────────────────────────────────────────┬────────────────────────────────────────┘
                                                ↓
                               [Tailored Wellness Activities]
                               • Ranked Recommendations with Scores
                               • Explainable Breakdown & Strategies
                               • Interactive Feedback Loop (Like / Dislike / Select)
```

---

## 📐 Formulas, Schemas & Specifications

### 1. Dynamic Emotional Intensity Formula (Task 1)
$$\text{Intensity} = \text{clamp}\left(0.45 \cdot P_{\text{dom}} + 0.25 \cdot \max(P_{\text{pos}}, P_{\text{neg}}) + 0.20 \cdot W_{\text{sev}}(E_{\text{dom}}) + 0.10 \cdot (1 - \bar{H}), 0.0, 1.0\right)$$
* $P_{\text{dom}}$: Dominant emotion confidence score.
* $P_{\text{pos}}, P_{\text{neg}}$: Positive and negative aggregate polarities.
* $W_{\text{sev}}$: Emotion severity category weight (Anger/Fear: 0.85, Sadness: 0.80, Disgust: 0.75, Surprise: 0.70, Joy: 0.65).
* $\bar{H}$: Normalized Shannon entropy of the probability distribution.

### 2. Severity Classification Tiers
* **Low:** $\text{Intensity} < 0.35$
* **Moderate:** $0.35 \le \text{Intensity} < 0.65$
* **High:** $0.65 \le \text{Intensity} < 0.85$
* **Very High:** $\text{Intensity} \ge 0.85$

### 3. Dynamic Multi-Factor Ranking Formula (Task 4)
$$\begin{aligned}
\text{Final Score} = & \; 0.30 \times \text{Emotion Relevance} \\
& + 0.15 \times \text{Intensity Fit} \\
& + 0.20 \times \text{User Preference Match} \\
& + 0.20 \times \text{Semantic Cosine Similarity} \\
& + 0.10 \times \text{Historical Preference Score} \\
& + 0.05 \times \text{Novelty Score} \\
& - \text{Duplicate Penalty} - \text{Low Relevance Penalty}
\end{aligned}$$

---

## 📁 Repository Structure

```text
sentiment_emotion_project/
│
├── app.py                     # Unified Streamlit Web Dashboard (Milestones 1, 2, 3)
├── app_milestone1.py          # Dedicated Milestone 1 App (Ingestion & VADER)
├── app_milestone2.py          # Dedicated Milestone 2 App (BERT/DistilBERT Emotion)
├── app_milestone3.py          # Dedicated Milestone 3 App (Wellness Recommendations)
│
├── verify_milestone1.py       # Milestone 1 CLI Runner
├── verify_milestone2.py       # Milestone 2 CLI Runner
├── verify_milestone3.py       # Milestone 3 CLI Runner
│
├── MILESTONE_1_GUIDE.md       # Detailed Milestone 1 Guide
├── MILESTONE_2_GUIDE.md       # Detailed Milestone 2 Guide
├── MILESTONE_3_GUIDE.md       # Detailed Milestone 3 Guide
├── README.md                  # Unified Project Documentation
├── requirements.txt           # Dependencies
│
├── services/
│   ├── config.py              # Central configurations, constants, paths & weights
│   ├── ingestion.py           # Multi-channel text ingestion & strict validation
│   ├── preprocessing.py       # Negation-preserving NLP cleaning & lemmatization
│   ├── sentiment.py           # VADER baseline sentiment analyzer
│   ├── dataset_loader.py      # PyTorch multi-label dataset & label normalizer
│   ├── model_training.py      # Pos-weighted BCE fine-tuning engine (BERT / DistilBERT)
│   ├── emotion.py             # Multi-label sigmoid inference engine
│   ├── evaluation.py          # Scikit-learn multi-label comparative evaluation
│   ├── isear_validation.py    # Held-out ISEAR benchmark evaluation runner
│   ├── intensity.py           # Task 1: Emotional intensity & state analyzer
│   ├── wellness_data.py       # Task 2: Wellness repository and dataset loader
│   ├── user_profile.py        # Task 2: User profile, preferences & feedback loop
│   ├── semantic_matcher.py    # Task 5: SentenceTransformer dense cosine matcher
│   ├── hybrid_recommender.py  # Task 3 & 4: Hybrid candidate generation & dynamic ranking
│   └── reporting.py           # Unified tabular reporting
│
├── scripts/
│   ├── prepare_datasets.py    # Dataset creation & ISEAR benchmark preparation
│   ├── train_bert.py          # Standalone BERT training script
│   ├── train_distilbert.py    # Standalone DistilBERT training script
│   ├── evaluate_models.py     # Standalone model evaluation & comparison script
│   ├── validate_isear.py      # Standalone ISEAR benchmark validation script
│   └── generate_embeddings.py # Precomputes and caches wellness dense embeddings
│
├── data/
│   ├── sample_corpus.csv      # Milestone 1 benchmark corpus
│   ├── train_emotions.csv     # 6-emotion multi-label training set
│   ├── val_emotions.csv       # 6-emotion validation set
│   ├── test_emotions.csv      # 6-emotion test set
│   ├── isear_benchmark.csv    # Held-out ISEAR benchmark dataset
│   ├── wellness_content.csv   # Curated wellness activities dataset
│   ├── user_profiles.json     # User profile and preference storage
│   └── interaction_history.json # Timestamped interaction & feedback storage
│
├── models/
│   ├── bert_emotion/          # Fine-tuned BERT model and tokenizer
│   ├── distilbert_emotion/    # Fine-tuned DistilBERT model and tokenizer
│   ├── wellness_embeddings.npy# Cached precomputed dense embeddings matrix
│   └── wellness_embeddings_meta.json # Embeddings metadata
│
└── tests/
    ├── test_ingestion.py      # 17 ingestion unit tests
    ├── test_preprocessing.py  # 11 preprocessing unit tests
    ├── test_sentiment.py      # 6 VADER sentiment unit tests
    ├── test_pipeline.py       # 7 Milestone 1 integration tests
    ├── test_emotion.py        # 10 Transformer emotion unit tests
    ├── test_evaluation.py     # 3 model evaluation unit tests
    ├── test_isear.py          # 2 ISEAR benchmark unit tests
    ├── test_integration_milestone2.py # 4 Milestone 2 integration tests
    ├── test_intensity.py      # 8 emotion intensity unit tests (Task 1)
    ├── test_user_profile.py   # 6 user profile & feedback unit tests (Task 2)
    ├── test_semantic.py       # 4 semantic embedding unit tests (Task 5)
    ├── test_hybrid_recommender.py # 5 hybrid ranking unit tests (Task 3 & 4)
    └── test_integration_milestone3.py # 3 Milestone 3 integration tests
```

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 2. Download NLTK Resources
```bash
python download_nltk.py
```

### 3. Generate Semantic Embeddings Cache
```bash
python scripts/generate_embeddings.py
```

---

## 🧪 Running Automated Tests

Run the complete 85-test suite across all three milestones:
```bash
pytest -q
```
**Test Results:** `85 passed (100% Pass Rate)`

---

## 💻 Running the Streamlit Applications

```bash
# 1. Run Unified Platform (Milestones 1, 2, and 3):
streamlit run app.py

# 2. Run Dedicated Milestone 1 App:
streamlit run app_milestone1.py

# 3. Run Dedicated Milestone 2 App:
streamlit run app_milestone2.py

# 4. Run Dedicated Milestone 3 App:
streamlit run app_milestone3.py
```
Open `http://localhost:8501` in your browser.

---

## 🔒 Privacy, Safety & Ethical Limitations
1. **No Medical Claims:** This platform is designed solely for self-care, reflection, and workplace wellness. It does not diagnose, treat, or assess psychiatric conditions.
2. **Data Minimization:** No personally identifiable information (PII) or sensitive health records are stored. Profiles track only interaction counters, liked tags, and emotional trend history.
3. **Transparent Fallbacks:** When insufficient interaction matrices exist for Collaborative Filtering, the system explicitly reports status and activates content/semantic fallbacks without generating mock data.
