# Milestone 1: Text Ingestion, NLP Preprocessing & Baseline VADER Sentiment

## 📌 Milestone 1 Summary
Milestone 1 implements an end-to-end NLP text preprocessing and baseline sentiment classification pipeline using NLTK and VADER.

---

## 🚀 How to Run Milestone 1

### 1. Launch Dedicated Milestone 1 Web Dashboard:
```bash
streamlit run app_milestone1.py
```

### 2. Run Milestone 1 Verification Script:
```bash
python verify_milestone1.py
```

### 3. Run Milestone 1 Automated Unit & Integration Tests:
```bash
# Run all Milestone 1 tests
pytest -v tests/test_ingestion.py tests/test_preprocessing.py tests/test_sentiment.py tests/test_pipeline.py
```

---

## 📁 Milestone 1 File Mapping

| Component | File Path | Description |
|---|---|---|
| **UI Dashboard** | [`app_milestone1.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/app_milestone1.py) | Streamlit web interface dedicated solely to Milestone 1. |
| **Verification Runner** | [`verify_milestone1.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/verify_milestone1.py) | Standalone command-line verification script. |
| **Ingestion Service** | [`services/ingestion.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/ingestion.py) | Direct text, TXT, and CSV file ingestion & strict validation. |
| **Preprocessing Service**| [`services/preprocessing.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/preprocessing.py) | Cleaning, tokenization, lemmatization & negation preservation (`not`, `no`, `never`). |
| **Sentiment Service** | [`services/sentiment.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/sentiment.py) | NLTK VADER sentiment analyzer & compound classification. |
| **Reporting Service** | [`services/reporting.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/reporting.py) | Tabular report generator & aggregate counters. |
| **Benchmark Corpus** | [`data/sample_corpus.csv`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/data/sample_corpus.csv) | Standard 9-sample benchmark corpus. |
| **Ingestion Tests** | [`tests/test_ingestion.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_ingestion.py) | 17 unit tests for text, TXT, and CSV validation. |
| **Preprocessing Tests** | [`tests/test_preprocessing.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_preprocessing.py) | 11 unit tests for cleaning, tokenization, lemmatization, and negations. |
| **Sentiment Tests** | [`tests/test_sentiment.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_sentiment.py) | 6 unit tests for VADER dynamic scoring and thresholds. |
| **Pipeline Tests** | [`tests/test_pipeline.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_pipeline.py) | 7 integration tests for the full Milestone 1 flow. |

---

## 📊 Benchmark Corpus Results (Milestone 1)

| Metric | Result |
|---|---|
| Total Samples Processed | 9 |
| Valid Samples | 9 |
| Invalid Samples | 0 |
| Positive Count | 4 |
| Negative Count | 3 |
| Neutral Count | 2 |
