# Text Sentiment Analysis & Deep Emotion Classification System

An end-to-end NLP and Deep Learning classification system integrating **Milestone 1 (Ingestion, Preprocessing & VADER Sentiment Baseline)** and **Milestone 2 (BERT & DistilBERT 6-Emotion Multi-Label Classification)** with comprehensive evaluation, ISEAR benchmark validation, and an interactive Streamlit UI.

---

## 🌟 Milestone 1 & 2 Key Features

1. **Multi-Channel Text Ingestion & Validation (Milestone 1)**
   - **Direct Text Input**, **TXT File Upload**, and **CSV Upload** (with `text` column).
   - Strict validation preventing empty strings, whitespace, corrupt headers, and invalid formats.

2. **Sentiment-Aware NLP Preprocessing (Milestone 1)**
   - Sanitization of URLs, HTML tags, mentions, and hashtags.
   - Contraction expansion (`can't` → `can not`, `won't` → `will not`).
   - **Negation Preservation**: Retains critical sentiment modifiers (`not`, `no`, `never`, `cannot`, etc.).
   - **WordNet Lemmatization**: Accurate POS-tagged base word normalization.

3. **VADER Baseline Sentiment Engine (Milestone 1)**
   - Granular scores: `pos`, `neg`, `neu`, and `compound`.
   - Dynamic classification thresholds: $\text{Compound} \ge 0.05$ (Positive), $\le -0.05$ (Negative), otherwise Neutral.

4. **Multi-Label Deep Emotion Classification (Milestone 2)**
   - **6 Core Ekman Emotion Categories**:
     1. Joy
     2. Sadness
     3. Anger
     4. Fear
     5. Surprise
     6. Disgust
   - **Transformer Architectures**: Pre-trained & fine-tuned `BERT` (`bert-base-uncased`) and `DistilBERT` (`distilbert-base-uncased`).
   - **True Multi-Label Probabilities**: Multi-label Sigmoid activations with pos-weighted `BCEWithLogitsLoss`.
   - **Configurable Confidence Thresholding**: Default $0.50$ (customizable $0.10 - 0.90$).
   - **Primary Emotion**: Highest-probability emotion dynamically identified.

5. **Model Evaluation & Comparative Analysis (Milestone 2)**
   - Multi-label evaluation: Subset Accuracy, Hamming Accuracy, Macro/Micro Precision, Recall, and Macro F1-Score via `scikit-learn`.
   - Side-by-side comparison report generated in `reports/model_comparison.json`.

6. **Held-Out ISEAR Benchmark Validation (Milestone 2)**
   - Validation against an independent, held-out subset of the International Survey on Emotion Antecedents and Reactions (ISEAR).
   - Generates emotion-wise metrics, accuracy, and error analyses saved to `reports/isear_results.csv` and `reports/isear_metrics.json`.

7. **Interactive Streamlit Web Dashboard**
   - Live text input, file uploaders, model selector (`BERT` / `DistilBERT`), threshold slider, progress bars for all 6 emotions, single-sample inspector, tabular reports, and benchmark report viewers.

---

## 📁 Project Architecture

```text
sentiment_emotion_project/
│
├── app.py                     # Streamlit web dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation & guide
├── verify_milestone1.py       # Milestone 1 CLI verification runner
├── verify_milestone2.py       # Milestone 2 CLI verification runner
│
├── services/
│   ├── __init__.py
│   ├── config.py              # Centralized constants, paths, labels, device setup
│   ├── ingestion.py           # Multi-channel text ingestion & strict validation
│   ├── preprocessing.py       # Text cleaning, tokenization, lemmatization & negation handling
│   ├── sentiment.py           # VADER baseline sentiment analyzer
│   ├── dataset_loader.py      # Multi-label PyTorch Dataset & CSV parsers
│   ├── model_training.py      # Fine-tuning engine for BERT and DistilBERT
│   ├── emotion.py             # Transformer inference engine & confidence scoring
│   ├── evaluation.py          # Multi-label scikit-learn evaluation & comparison
│   ├── isear_validation.py    # Held-out ISEAR benchmark evaluation runner
│   └── reporting.py           # Integrated Milestone 1 + 2 report generator
│
├── scripts/
│   ├── prepare_datasets.py    # Dataset creation & ISEAR benchmark preparation
│   ├── train_bert.py          # Standalone BERT training script
│   ├── train_distilbert.py    # Standalone DistilBERT training script
│   ├── evaluate_models.py     # Standalone model evaluation & comparison script
│   └── validate_isear.py      # Standalone ISEAR benchmark validation script
│
├── models/
│   ├── bert_emotion/          # Saved fine-tuned BERT model and tokenizer
│   └── distilbert_emotion/    # Saved fine-tuned DistilBERT model and tokenizer
│
├── data/
│   ├── sample_corpus.csv      # Milestone 1 benchmark corpus
│   ├── train_emotions.csv     # 6-emotion multi-label training set
│   ├── val_emotions.csv       # 6-emotion validation set
│   ├── test_emotions.csv      # 6-emotion test set
│   └── isear_benchmark.csv    # Held-out ISEAR benchmark dataset
│
├── reports/
│   ├── bert_metrics.json      # BERT test evaluation metrics
│   ├── distilbert_metrics.json# DistilBERT test evaluation metrics
│   ├── model_comparison.json  # Side-by-side BERT vs DistilBERT comparison
│   ├── isear_metrics.json     # ISEAR benchmark validation metrics
│   └── isear_results.csv      # Sample-by-sample ISEAR predictions and error report
│
└── tests/
    ├── __init__.py
    ├── test_ingestion.py      # 17 ingestion unit tests
    ├── test_preprocessing.py  # 11 preprocessing unit tests
    ├── test_sentiment.py      # 6 VADER sentiment unit tests
    ├── test_pipeline.py       # 7 Milestone 1 integration tests
    ├── test_emotion.py        # 10 Transformer emotion unit tests
    ├── test_evaluation.py     # 3 model evaluation unit tests
    ├── test_isear.py          # 2 ISEAR benchmark unit tests
    └── test_integration_milestone2.py # 4 Milestone 2 end-to-end integration tests
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

---

## 🏋️ Training & Validation Commands

### Train DistilBERT:
```bash
python scripts/train_distilbert.py
```

### Train BERT:
```bash
python scripts/train_bert.py
```

### Evaluate & Compare Models:
```bash
python scripts/evaluate_models.py
```

### Validate on Held-Out ISEAR Benchmark:
```bash
python scripts/validate_isear.py
```

---

## 🧪 Running Automated Tests

Run the complete 59-test suite:
```bash
pytest -v
```

---

## 💻 Running the Streamlit Application

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.
