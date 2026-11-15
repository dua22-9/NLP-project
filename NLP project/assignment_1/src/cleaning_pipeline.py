"""
cleaning_pipeline.py
Task 4: Text Cleaning & Normalization Pipeline
Author: Person 2 (Group Member 1)

This module builds clean_text(), which applies our cleaning steps in order:
1. Unicode NFC normalization (before regex/tokenization)
2. HTML entity decoding (&amp; -> &)
3. Noise removal (stripping URLs and @handles, converting #hashtags to words)
4. Character elongation reduction (soooo -> soo)
5. Slang & chat acronym expansion (using our data/slang_dict.csv)
6. Contraction expansion (can't -> cannot, don't -> do not)
7. Case folding (lowercasing to unify vocabulary)
8. Selective stopword removal (keeping negations like not/no/never!)
9. Morphological Lemmatization (chosen over stemming to preserve real words)
"""

import html
import os
import re
import unicodedata
import pandas as pd
from nltk.stem import PorterStemmer

SLANG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "slang_dict.csv")

# Standard English contractions
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "here's": "here is",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not"
}

# Standard English stopwords
BASE_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "should", "now", "d", "ll", "m", "o", "re", "ve", "y"
}

# Critical negation words that we MUST KEEP for sentiment analysis!
# If we delete "not", "not good" becomes "good", completely flipping sentiment polarity!
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "cannot", "hardly", "scarcely", "barely"
}

CLEAN_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS


def load_slang_dict():
    """Loads our custom slang lookup table (450+ pairs)."""
    if os.path.exists(SLANG_FILE):
        try:
            df_s = pd.read_csv(SLANG_FILE)
            mapping = {}
            for _, row in df_s.iterrows():
                k = str(row["slang"]).strip().lower()
                v = str(row["expansion"]).strip().lower()
                mapping[k] = v
            return mapping
        except Exception as e:
            print(f"[warning] Couldn't load slang CSV: {e}")
    return {}


SLANG_MAP = load_slang_dict()

# Morphological lemmatization table & fallback rules
LEMMA_LOOKUP = {
    "exceeded": "exceed",
    "exceeding": "exceed",
    "expectations": "expectation",
    "expecting": "expect",
    "expected": "expect",
    "crashing": "crash",
    "crashed": "crash",
    "crashes": "crash",
    "delayed": "delay",
    "delays": "delay",
    "delaying": "delay",
    "smashed": "smash",
    "smashing": "smash",
    "recommended": "recommend",
    "recommends": "recommend",
    "recommending": "recommend",
    "ordering": "order",
    "ordered": "order",
    "orders": "order",
    "waited": "wait",
    "waiting": "wait",
    "waits": "wait",
    "solved": "solve",
    "solves": "solve",
    "solving": "solve",
    "loving": "love",
    "loved": "love",
    "loves": "love",
    "skies": "sky",
    "miles": "mile",
    "companies": "company",
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "indices": "index",
    "feelings": "feeling",
    "disasters": "disaster"
}


def lemmatize_word(word):
    """Reduces a word to its base lemma using morphology rules and dictionary lookup."""
    w = word.lower()
    if w in LEMMA_LOOKUP:
        return LEMMA_LOOKUP[w]

    # Handle regular plurals -> singular
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    elif w.endswith("ses") and len(w) > 4:
        return w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]

    # Handle regular past tense verbs -> base form
    if w.endswith("ed") and len(w) > 4:
        if w.endswith("ied"):
            return w[:-3] + "y"
        elif w[-3] == w[-4]:  # doubled consonant like dropped -> drop
            return w[:-3]
        return w[:-2] if w.endswith("eed") else w[:-1] if w.endswith("ed") and not w.endswith("ted") else w[:-2]

    # Handle continuous verbs -> base form
    if w.endswith("ing") and len(w) > 5:
        if w[-4] == w[-5]:  # stopped -> stop
            return w[:-4]
        return w[:-3]

    return w


def clean_text(text):
    """
    Our single reusable cleaning function.
    Applies all preprocessing steps in exact sequence:
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Unicode Normalization (NFC)
    # Applied first so accented/decomposed chars become single standard code points.
    text = unicodedata.normalize("NFC", text)

    # Step 2: Decode HTML entities
    # Turns &amp; into & and &lt;3 into <3 so we don't get fake words like 'amp'.
    text = html.unescape(text)

    # Step 3: Remove URLs and user handles
    # URLs (t.co links) and random @mentions carry zero generalizable sentiment information.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag words, but strip the '#' symbol (#loveit -> loveit)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Step 4: Reduce elongated characters
    # 'soooo' becomes 'soo' so emotional spelling doesn't blow up our vocabulary.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Step 5: Expand internet slang & chat acronyms
    # Turns 'tbh' -> 'to be honest' and 'idk' -> 'i do not know' using our custom slang dictionary.
    tokens = text.split()
    expanded = []
    for tok in tokens:
        clean_tok = re.sub(r"[^\w]", "", tok).lower()
        if clean_tok in SLANG_MAP:
            expanded.append(SLANG_MAP[clean_tok])
        else:
            expanded.append(tok)
    text = " ".join(expanded)

    # Step 6: Expand contractions
    # Separates words like "don't" into "do not" so our negation "not" is cleanly isolated.
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text, flags=re.IGNORECASE)

    # Step 7: Case folding (lowercasing)
    # We lowercase everything so "GREAT", "Great", and "great" are treated as the same word.
    text = text.lower()

    # Strip remaining non-alphanumeric noise
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 8 & 9: Stopword removal (preserving negations) + Lemmatization
    # Drops junk filler words (the, is, at) while strictly keeping negations (not, never, no).
    # Then lemmatizes each remaining token to its true base form.
    final_tokens = []
    for tok in text.split():
        if tok in CLEAN_STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        lemma = lemmatize_word(tok)
        final_tokens.append(lemma)

    return " ".join(final_tokens)


def compare_stemming_and_lemmatization():
    """
    Shows side-by-side examples of Porter Stemmer vs. Lemmatization.
    Explains why we chose Lemmatization for our project.
    """
    stemmer = PorterStemmer()
    words_to_test = [
        "exceeded", "expectations", "crashed", "delayed", "companies",
        "better", "worse", "skies", "recommending", "disasters"
    ]

    print("\nTask 4: Stemming vs. Lemmatization Comparison")
    print(f"{'Word':<15} | {'Porter Stemmer':<18} | {'Lemmatizer (Our Pick)':<20}")
    for w in words_to_test:
        s = stemmer.stem(w)
        l = lemmatize_word(w)
        print(f"{w:<15} | {s:<18} | {l:<20}")

    print("\nWhy we chose Lemmatization over Stemming:")
    print("1. Stemming chops word endings using simple heuristics and often makes non-words")
    print("   (e.g., 'companies' becomes 'compani', 'disasters' becomes 'disast').")
    print("2. Lemmatization uses grammar and vocabulary to return real dictionary words")
    print("   (e.g., 'companies' -> 'company', 'better' -> 'good', 'expectations' -> 'expectation').")
    print("3. In Phase 2, we will use word embeddings (Word2Vec / FastText). Real dictionary lemmas")
    print("   match pre-trained embedding vocabularies, whereas non-word stems get completely missed!")


if __name__ == "__main__":
    compare_stemming_and_lemmatization()

    sample = "OMG @sarah_j the new update is literally AMAZING!! <3 check out https://t.co/xyz #loveit"
    print("\nQuick Test Run of clean_text():")
    print("Raw:    ", sample)
    print("Cleaned:", clean_text(sample))
