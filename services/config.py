"""
Configuration module for Emotion Classification and Sentiment Analysis.
Contains constants, emotion labels, paths, device configuration, and default parameters.
"""

import os
from pathlib import Path
import ssl
from typing import List
import urllib3
import torch

# Handle local Windows SSL certificate environments gracefully
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def patch_network_ssl():
    """Patches requests, urllib3, and httpx to bypass local SSL certificate validation issues."""
    try:
        import httpx
        _old_httpx_init = httpx.Client.__init__

        def _patched_httpx_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _old_httpx_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_httpx_init
    except Exception:
        pass

    try:
        import requests
        _old_req_init = requests.Session.request

        def _patched_req_init(self, *args, **kwargs):
            kwargs["verify"] = False
            return _old_req_init(self, *args, **kwargs)

        requests.Session.request = _patched_req_init
    except Exception:
        pass


patch_network_ssl()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Exactly the 6 required emotion categories
EMOTIONS: List[str] = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
NUM_EMOTIONS: int = len(EMOTIONS)
EMOTION2ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
ID2EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

# Capitalized display names for UI/Reporting
EMOTION_DISPLAY_NAMES = {
    "joy": "Joy",
    "sadness": "Sadness",
    "anger": "Anger",
    "fear": "Fear",
    "surprise": "Surprise",
    "disgust": "Disgust",
}

# Directories and paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

BERT_MODEL_DIR = MODELS_DIR / "bert_emotion"
DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_emotion"

TRAIN_DATA_PATH = DATA_DIR / "train_emotions.csv"
VAL_DATA_PATH = DATA_DIR / "val_emotions.csv"
TEST_DATA_PATH = DATA_DIR / "test_emotions.csv"
ISEAR_BENCHMARK_PATH = DATA_DIR / "isear_benchmark.csv"

BERT_METRICS_PATH = REPORTS_DIR / "bert_metrics.json"
DISTILBERT_METRICS_PATH = REPORTS_DIR / "distilbert_metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"
ISEAR_RESULTS_PATH = REPORTS_DIR / "isear_results.csv"
ISEAR_METRICS_PATH = REPORTS_DIR / "isear_metrics.json"

# Milestone 3 - Wellness and Profile Paths
WELLNESS_CONTENT_PATH = DATA_DIR / "wellness_content.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.json"
INTERACTION_HISTORY_PATH = DATA_DIR / "interaction_history.json"
FEEDBACK_EVENTS_PATH = DATA_DIR / "feedback_events.json"
WELLNESS_EMBEDDINGS_PATH = MODELS_DIR / "wellness_embeddings.npy"
WELLNESS_EMBEDDINGS_META_PATH = MODELS_DIR / "wellness_embeddings_meta.json"

# Base model architectures & embeddings
BERT_BASE_MODEL = "bert-base-uncased"
DISTILBERT_BASE_MODEL = "distilbert-base-uncased"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Default hyper-parameters & settings
DEFAULT_THRESHOLD = 0.50
DEFAULT_MIXED_THRESHOLD = 0.30
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 4

# Severity levels & thresholds
SEVERITY_THRESHOLDS = {
    "Low": 0.35,
    "Moderate": 0.65,
    "High": 0.85,
}

# Emotion category severity base weights for intensity formula
EMOTION_SEVERITY_WEIGHTS = {
    "anger": 0.85,
    "fear": 0.85,
    "sadness": 0.80,
    "disgust": 0.75,
    "surprise": 0.70,
    "joy": 0.65,
}

# Ranking Weights (Sum of positive components = 1.0)
DEFAULT_RANKING_WEIGHTS = {
    "emotion_weight": 0.30,
    "intensity_weight": 0.15,
    "preference_weight": 0.20,
    "similarity_weight": 0.20,
    "history_weight": 0.10,
    "novelty_weight": 0.05,
    "duplicate_penalty": 0.20,
    "low_relevance_penalty": 0.30,
}

# Medical Disclaimer
MEDICAL_DISCLAIMER = (
    "⚠️ Medical Disclaimer: This system estimates emotional states from text and recommends "
    "wellness content. It is not a medical diagnostic or clinical assessment system."
)


def get_device() -> torch.device:
    """Returns CUDA device if GPU is available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

