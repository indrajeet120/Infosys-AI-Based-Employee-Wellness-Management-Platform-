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

# Base model architectures
BERT_BASE_MODEL = "bert-base-uncased"
DISTILBERT_BASE_MODEL = "distilbert-base-uncased"

# Default hyper-parameters & settings
DEFAULT_THRESHOLD = 0.50
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 4


def get_device() -> torch.device:
    """Returns CUDA device if GPU is available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
