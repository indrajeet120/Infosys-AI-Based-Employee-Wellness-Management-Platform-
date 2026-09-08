"""
Preprocessing service for Text Sentiment Analysis.
Performs text normalization, noise filtering, special-character & punctuation handling,
tokenization, negation-preserving stopword removal, and lemmatization.
"""

import re
import string
from typing import List, Optional, Set
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sentiment-critical negation and modifier words that must NOT be removed as stopwords
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "none", "neither", "nor", "nothing", "nobody",
    "nowhere", "hardly", "scarcely", "barely", "cannot", "n't", "cant",
    "wont", "dont", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "doesnt", "didnt", "shouldnt", "couldnt", "wouldnt", "without",
    "rarely", "seldom", "despite", "against"
}

_RESOURCES_INITIALIZED = False


def ensure_nltk_resources() -> None:
    """Downloads required NLTK resources if not already present."""
    global _RESOURCES_INITIALIZED
    if _RESOURCES_INITIALIZED:
        return

    required_resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("corpora/stopwords.zip", "stopwords"),
        ("tokenizers/punkt.zip", "punkt"),
        ("tokenizers/punkt_tab.zip", "punkt_tab"),
        ("corpora/wordnet.zip", "wordnet"),
        ("corpora/omw-1.4.zip", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger.zip", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng.zip", "averaged_perceptron_tagger_eng"),
    ]

    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except (LookupError, IndexError):
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass

    _RESOURCES_INITIALIZED = True


def get_wordnet_pos(nltk_tag: str) -> str:
    """Map NLTK POS tag to WordNet POS tag."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class TextPreprocessor:
    """
    NLP Preprocessor for sentiment analysis tasks.
    Cleans, tokenizes, removes non-sentiment stopwords, and lemmatizes text.
    """

    def __init__(self, preserve_negations: bool = True):
        ensure_nltk_resources()
        self.preserve_negations = preserve_negations
        self.lemmatizer = WordNetLemmatizer()

        try:
            base_stopwords = set(stopwords.words("english"))
        except Exception:
            base_stopwords = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                "your", "yours", "yourself", "yourselves", "he", "him", "his",
                "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "what", "which",
                "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and",
                "but", "if", "or", "because", "as", "until", "while", "of", "at",
                "by", "for", "with", "about", "between", "into", "through",
                "during", "before", "after", "above", "below", "to", "from",
                "up", "down", "in", "out", "on", "off", "over", "under", "again",
                "further", "then", "once", "here", "there", "when", "where",
                "why", "how", "all", "any", "both", "each", "few", "more", "most",
                "other", "some", "such", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "should", "now"
            }

        if self.preserve_negations:
            # Remove negation words from stop words set
            self.stop_words = base_stopwords - NEGATION_WORDS
        else:
            self.stop_words = base_stopwords

    def clean_text(self, text: str) -> str:
        """
        Removes URLs, HTML tags, special symbols, and normalizes spaces.
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # Handle mentions and hashtags: preserve the word part (e.g. #happy -> happy, @product -> product)
        text = re.sub(r"[@#](\w+)", r" \1 ", text)

        # Replace contractions (e.g., n't -> not)
        text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bn't\b", " not", text, flags=re.IGNORECASE)

        # Replace non-alphanumeric chars (excluding standard spaces) with spaces
        # Note: preserve words and numbers
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Normalize repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into a list of word tokens."""
        if not text or not text.strip():
            return []
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Removes stopwords while strictly preserving negation tokens."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 0]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatizes tokens with POS tag mapping."""
        if not tokens:
            return []
        try:
            pos_tags = nltk.pos_tag(tokens)
            lemmatized = [
                self.lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                for word, tag in pos_tags
            ]
            return lemmatized
        except Exception:
            # Fallback if pos_tag fails
            return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess(self, text: Optional[str]) -> str:
        """
        Executes the full preprocessing pipeline on input text.

        Returns:
            Processed and cleaned text string.
        """
        if text is None:
            return ""

        # Step 1-5: Normalization, noise filtering, special char handling, repeated spaces
        cleaned = self.clean_text(str(text))
        if not cleaned:
            return ""

        # Step 6: Tokenization
        tokens = self.tokenize(cleaned)

        # Step 7: Stopword removal (preserving negations)
        filtered_tokens = self.remove_stopwords(tokens)

        # Step 8: Lemmatization
        lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)

        return " ".join(lemmatized_tokens)


# Singleton preprocessor instance for convenience
_default_preprocessor: Optional[TextPreprocessor] = None


def preprocess_text(text: Optional[str], preserve_negations: bool = True) -> str:
    """
    Convenience function for preprocessing a single text string.
    """
    global _default_preprocessor
    if _default_preprocessor is None or _default_preprocessor.preserve_negations != preserve_negations:
        _default_preprocessor = TextPreprocessor(preserve_negations=preserve_negations)
    return _default_preprocessor.preprocess(text)
