"""
Script to download and verify all required NLTK corpora, tokenizers, and models:
- punkt / punkt_tab (for word_tokenize)
- stopwords (for stopword removal)
- wordnet / omw-1.4 (for WordNetLemmatizer & synsets)
- averaged_perceptron_tagger (for POS tagging)
- vader_lexicon (for VADER SentimentIntensityAnalyzer)
"""

import ssl
import nltk

# Handle SSL certificate verification if needed in local corporate/school environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


RESOURCES = [
    "vader_lexicon",
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def setup_nltk():
    print("Downloading all required NLTK resources...")
    for resource in RESOURCES:
        try:
            print(f"-> Downloading '{resource}'...")
            nltk.download(resource, quiet=False)
        except Exception as e:
            print(f"Warning: Failed to download '{resource}': {e}")

    print("\nVerifying NLTK modules and imports...")
    try:
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        # Verify stopwords
        sw = stopwords.words("english")
        print(f"✓ stopwords verified ({len(sw)} English stopwords loaded)")

        # Verify wordnet
        syns = wordnet.synsets("happy")
        print(f"✓ wordnet verified ({len(syns)} synsets found for 'happy')")

        # Verify lemmatizer
        lem = WordNetLemmatizer()
        res = lem.lemmatize("running", "v")
        print(f"✓ WordNetLemmatizer verified ('running' (verb) -> '{res}')")

        # Verify word_tokenize
        tokens = word_tokenize("I love natural language processing!")
        print(f"✓ word_tokenize verified (tokenized into: {tokens})")

        # Verify VADER
        vader = SentimentIntensityAnalyzer()
        scores = vader.polarity_scores("Great job!")
        print(f"✓ VADER SentimentIntensityAnalyzer verified (compound: {scores['compound']})")

        print("\nSUCCESS: All NLTK packages and resources are installed and verified!")

    except Exception as e:
        print(f"\nERROR: Verification failed: {e}")


if __name__ == "__main__":
    setup_nltk()
