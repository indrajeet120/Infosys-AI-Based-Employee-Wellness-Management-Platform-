# Milestone 3: Advanced Emotion Intensity Analysis & Personalized Wellness Recommendation System

## 📌 Milestone 3 Summary
Milestone 3 upgrades the system from sentiment and emotion detection into an **AI-driven Personalized Wellness Recommendation System**. It calculates dynamic emotion intensity, positive/negative polarity, mixed emotional states, and severity tiers, feeding into a hybrid recommendation engine combining rule-based, content-based, preference-based, emotion similarity, and dense Transformer semantic matching with interactive feedback learning.

---

> ⚠️ **Medical Disclaimer:**  
> *This system estimates emotional states from text and recommends wellness content. It is not a medical diagnostic or clinical assessment system.*

---

## 🚀 How to Run Milestone 3

### 1. Launch Dedicated Milestone 3 Web Dashboard:
```bash
streamlit run app_milestone3.py
```

### 2. Run Milestone 3 CLI Verification Runner:
```bash
python verify_milestone3.py
```

### 3. Precompute / Refresh Semantic Content Embeddings:
```bash
python scripts/generate_embeddings.py
```

### 4. Run Milestone 3 Automated Unit & Integration Tests:
```bash
# Run Milestone 3 test suite (26 tests)
pytest tests/test_intensity.py tests/test_user_profile.py tests/test_semantic.py tests/test_hybrid_recommender.py tests/test_integration_milestone3.py -v

# Run complete 85-test suite across Milestones 1, 2, and 3:
pytest -v
```

---

## 🏗️ Milestone 3 System Architecture

```text
[User Text Input]
       ↓
[Fine-Tuned BERT / DistilBERT]
       ↓
[6-Emotion Probabilities (Sigmoid)]
       ↓
[Intensity & Emotional State Engine]
• Dominant Emotion & Confidence
• Emotional Intensity (0.0 to 1.0)
• Positive vs Negative Polarity
• Mixed Emotion Detection
• Severity Tier (Low, Moderate, High, Very High)
• Final Emotional State
       ↓
┌────────────────────────────────────────────────────────────────────────┐
│                      HYBRID RECOMMENDATION ENGINE                      │
│                                                                        │
│   Candidate Generation & Scoring:                                      │
│   1. Rule-Based Intensity Fit (Acute calming vs reflection)           │
│   2. Content-Based Tag Matching                                        │
│   3. User Preference Matching (Content type, activity, language)       │
│   4. Emotion Similarity (Target emotions ↔ Detected probabilities)     │
│   5. Dense Semantic Matching (Sentence Transformer Cosine Similarity)  │
│   6. Historical Behavior & Feedback Loop (Like boost / Dislike penalty)│
│                                                                        │
│   Dynamic Ranking Formula:                                             │
│   Final Score = w_emo*S_emo + w_int*S_int + w_pref*S_pref               │
│               + w_sim*S_sim + w_hist*S_hist + w_nov*S_nov              │
│               - Duplicate_Pen - Low_Relevance_Pen                      │
└────────────────────────────────────────────────────────────────────────┘
       ↓
[Explainable Ranked Recommendations]
• Final Score & Sub-scores
• Source Strategies Identified
• Dynamic Human-Readable Reason
• Interactive Feedback (Like, Dislike, Select, Skip)
```

---

## 📐 Formulas & Specifications

### 1. Dynamic Emotional Intensity Formula
$$\text{Intensity} = \text{clamp}\left(0.45 \cdot P_{\text{dom}} + 0.25 \cdot \max(P_{\text{pos}}, P_{\text{neg}}) + 0.20 \cdot W_{\text{sev}}(E_{\text{dom}}) + 0.10 \cdot (1 - \bar{H}), 0.0, 1.0\right)$$
* $P_{\text{dom}}$: Dominant emotion probability.
* $P_{\text{pos}}, P_{\text{neg}}$: Positive and negative polarity aggregates.
* $W_{\text{sev}}$: Emotion severity category weight (Anger/Fear: 0.85, Sadness: 0.80, Disgust: 0.75, Surprise: 0.70, Joy: 0.65).
* $\bar{H}$: Normalized Shannon entropy of the emotion probability distribution.

### 2. Severity Classification Tiers
* **Low:** $\text{Intensity} < 0.35$
* **Moderate:** $0.35 \le \text{Intensity} < 0.65$
* **High:** $0.65 \le \text{Intensity} < 0.85$
* **Very High:** $\text{Intensity} \ge 0.85$

### 3. Dynamic Hybrid Ranking Formula
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

## 📁 Milestone 3 File Mapping

| Component | File Path | Description |
|---|---|---|
| **UI Dashboard** | [`app_milestone3.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/app_milestone3.py) | Dedicated interactive Streamlit app for Milestone 3 recommendations & feedback. |
| **Verification Script**| [`verify_milestone3.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/verify_milestone3.py) | Standalone verification runner testing all 5 tasks. |
| **Intensity Engine** | [`services/intensity.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/intensity.py) | Task 1: Emotional state analysis, dynamic intensity, polarity, and severity. |
| **Wellness Dataset** | [`data/wellness_content.csv`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/data/wellness_content.csv) | Curated wellness activities dataset with 24 multi-modal activities. |
| **Wellness Loader** | [`services/wellness_data.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/wellness_data.py) | Task 2: Wellness repository loader and data models. |
| **Profile & Storage** | [`services/user_profile.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/user_profile.py) | Task 2: User profile manager, preferences, history tracking & feedback loop. |
| **Semantic Matcher** | [`services/semantic_matcher.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/semantic_matcher.py) | Task 5: SentenceTransformer dense embeddings and cosine similarity. |
| **Embedding Script** | [`scripts/generate_embeddings.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/scripts/generate_embeddings.py) | Precomputes and caches embeddings to `models/wellness_embeddings.npy`. |
| **Hybrid Engine** | [`services/hybrid_recommender.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/hybrid_recommender.py) | Task 3 & 4: Multi-strategy candidate generation, dynamic ranking & explanation. |
| **Intensity Tests** | [`tests/test_intensity.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_intensity.py) | 8 unit tests for Task 1. |
| **Profile Tests** | [`tests/test_user_profile.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_user_profile.py) | 6 unit tests for Task 2. |
| **Semantic Tests** | [`tests/test_semantic.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_semantic.py) | 4 unit tests for Task 5. |
| **Recommender Tests**| [`tests/test_hybrid_recommender.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_hybrid_recommender.py) | 5 unit tests for Task 3 & 4. |
| **Integration Tests**| [`tests/test_integration_milestone3.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_integration_milestone3.py) | 3 end-to-end integration tests. |
