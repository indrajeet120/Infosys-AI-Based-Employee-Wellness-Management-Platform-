# Milestone 2: Multi-Label Deep Emotion Classification & Benchmark Validation

## 📌 Milestone 2 Summary
Milestone 2 adds deep learning multi-label emotion classification across the 6 core emotions (**Joy, Sadness, Anger, Fear, Surprise, Disgust**) using fine-tuned **BERT** (`bert-base-uncased`) and **DistilBERT** (`distilbert-base-uncased`), model comparative evaluation, and held-out ISEAR benchmark validation.

---

## 🚀 How to Run Milestone 2

### 1. Launch Dedicated Milestone 2 Web Dashboard:
```bash
streamlit run app_milestone2.py
```

### 2. Run Milestone 2 Verification Script:
```bash
python verify_milestone2.py
```

### 3. Run Standalone Training & Evaluation Scripts:
```bash
# Train DistilBERT
python scripts/train_distilbert.py

# Train BERT
python scripts/train_bert.py

# Evaluate and Compare Models
python scripts/evaluate_models.py

# Validate on Held-Out ISEAR Benchmark
python scripts/validate_isear.py
```

### 4. Run Milestone 2 Automated Unit & Integration Tests:
```bash
# Run all Milestone 2 tests
pytest -v tests/test_emotion.py tests/test_evaluation.py tests/test_isear.py tests/test_integration_milestone2.py
```

---

## 📁 Milestone 2 File Mapping

| Component | File Path | Description |
|---|---|---|
| **UI Dashboard** | [`app_milestone2.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/app_milestone2.py) | Streamlit interface dedicated to Milestone 2 & emotion inspection. |
| **Verification Runner** | [`verify_milestone2.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/verify_milestone2.py) | Standalone command-line verification script for Milestone 2. |
| **Config Service** | [`services/config.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/config.py) | Emotion definitions, model paths, device detection, and parameters. |
| **Dataset Loader** | [`services/dataset_loader.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/dataset_loader.py) | Multi-label PyTorch Dataset & emotion label normalizer. |
| **Training Service** | [`services/model_training.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/model_training.py) | Fine-tuning orchestrator with pos-weighted BCE loss. |
| **Inference Service** | [`services/emotion.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/emotion.py) | Multi-label sigmoid probability computation & confidence scoring. |
| **Evaluation Service** | [`services/evaluation.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/evaluation.py) | Scikit-learn multi-label metrics (Accuracy, Precision, Recall, Macro F1). |
| **ISEAR Validation** | [`services/isear_validation.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/services/isear_validation.py) | Held-out ISEAR benchmark runner & error breakdown. |
| **Training Scripts** | [`scripts/`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/scripts) | `train_bert.py`, `train_distilbert.py`, `evaluate_models.py`, `validate_isear.py`. |
| **Saved Models** | [`models/`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/models) | Fine-tuned `bert_emotion/` and `distilbert_emotion/`. |
| **Generated Reports** | [`reports/`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/reports) | `bert_metrics.json`, `distilbert_metrics.json`, `model_comparison.json`, `isear_results.csv`, `isear_metrics.json`. |
| **Emotion Tests** | [`tests/test_emotion.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_emotion.py) | 10 unit tests for model loading, tokenization, single/multi-label inference. |
| **Evaluation Tests** | [`tests/test_evaluation.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_evaluation.py) | 3 unit tests for evaluation metrics and comparative reporting. |
| **ISEAR Tests** | [`tests/test_isear.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_isear.py) | 2 unit tests for ISEAR benchmark execution. |
| **Integration Tests** | [`tests/test_integration_milestone2.py`](file:///c:/Users/Indrajeet%20Yadav/OneDrive/Desktop/New%20folder/tests/test_integration_milestone2.py) | 4 end-to-end integration tests for the full Milestone 1 + 2 pipeline. |

---

## 📊 Model Comparison Results (Milestone 2)

| Metric | BERT (`bert-base-uncased`) | DistilBERT (`distilbert-base-uncased`) |
|---|---|---|
| **Accuracy (Hamming)** | **0.9000** | **0.9000** |
| **Exact Match Ratio** | **0.5000** | **0.5000** |
| **Precision (Macro)** | 0.7500 | **0.7778** |
| **Recall (Macro)** | 0.9444 | **1.0000** |
| **Macro F1-Score** | 0.8111 | **0.8536 (Winner 🏆)** |
| **Model Efficiency** | 110M params | **40% smaller, ~2x faster** |
