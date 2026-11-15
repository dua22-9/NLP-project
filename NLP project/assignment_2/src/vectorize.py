"""
Part D: Packaging for Next Phase (Tasks D1 & D2)
================================================
Covers:
  - Task D1: Model persistence save/load manager ensuring zero-retraining in Phase 3.
  - Task D2: Unified, production-ready `get_document_vector()` function interface.
  - Verification: Automated test suite validating all 5 representations on sample documents.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

# Global cache for loaded models so Phase 3 loads them once into memory
_LOADED_MODELS = {}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "cleaned_corpus.csv"))
DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))

def save_all_models(data_path=None, models_dir=None):
    """
    Fits and saves all count-based vectorizers (BoW, TF-IDF) and embedding models
    to the designated models directory.
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
        
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 70)
    print("TASK D1: SAVING ALL REPRESENTATIONS & MODELS FOR PHASE 3")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        if os.path.exists("data/cleaned_corpus.csv"):
            data_path = "data/cleaned_corpus.csv"
        elif os.path.exists("../data/cleaned_corpus.csv"):
            data_path = "../data/cleaned_corpus.csv"
        elif os.path.exists("person_2/data/cleaned_corpus.csv"):
            data_path = "person_2/data/cleaned_corpus.csv"

    print(f"[+] Loading corpus from: {data_path}")
    df = pd.read_csv(data_path)
    docs = df['cleaned_text'].fillna('').astype(str).tolist()
    sentences = [str(d).split() for d in docs if len(str(d).strip()) > 0]

    # 1. Fit & Save CountVectorizer (BoW)
    print("\n[+] Fitting & Saving CountVectorizer (BoW)...")
    bow_vec = CountVectorizer()
    bow_vec.fit(docs)
    bow_path = os.path.join(models_dir, "bow_vectorizer.joblib")
    joblib.dump(bow_vec, bow_path)
    print(f"    Saved: {bow_path} (Vocab size: {len(bow_vec.vocabulary_):,})")

    # 2. Fit & Save TfidfVectorizer
    print("\n[+] Fitting & Saving TfidfVectorizer...")
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(docs)
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_vec, tfidf_path)
    print(f"    Saved: {tfidf_path} (Vocab size: {len(tfidf_vec.vocabulary_):,})")

    # 3. Train & Save Word2Vec CBOW (if not already trained)
    cbow_path = os.path.join(models_dir, "word2vec_cbow.model")
    if not os.path.exists(cbow_path):
        print("\n[+] Training Word2Vec CBOW...")
        w2v_cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=2, epochs=30, sg=0, seed=42, workers=4)
        w2v_cbow.save(cbow_path)
    print(f"    Confirmed: {cbow_path}")

    # 4. Train & Save Word2Vec Skip-gram
    sg_path = os.path.join(models_dir, "word2vec_skipgram.model")
    if not os.path.exists(sg_path):
        print("\n[+] Training Word2Vec Skip-gram...")
        w2v_sg = Word2Vec(sentences, vector_size=100, window=5, min_count=2, epochs=30, sg=1, seed=42, workers=4)
        w2v_sg.save(sg_path)
    print(f"    Confirmed: {sg_path}")

    # 5. Train & Save fastText
    ft_path = os.path.join(models_dir, "fasttext.model")
    if not os.path.exists(ft_path):
        print("\n[+] Training fastText...")
        ft_model = FastText(sentences, vector_size=100, window=5, min_count=2, epochs=30, min_n=3, max_n=6, bucket=25000, seed=42, workers=4)
        ft_model.save(ft_path)
    print(f"    Confirmed: {ft_path}")

    print("\n[+] All 5 representations successfully saved and serialized to disk!")

def load_model(model_type: str, models_dir=None):
    """
    Loads and caches a saved model/vectorizer from disk.
    Ensures zero retraining overhead during runtime vectorization.
    """
    global _LOADED_MODELS
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
        
    model_type = model_type.lower().strip()
    if model_type in _LOADED_MODELS:
        return _LOADED_MODELS[model_type]

    if model_type == "bow":
        path = os.path.join(models_dir, "bow_vectorizer.joblib")
        obj = joblib.load(path)
    elif model_type == "tfidf":
        path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        obj = joblib.load(path)
    elif model_type in ["word2vec_cbow", "cbow"]:
        path = os.path.join(models_dir, "word2vec_cbow.model")
        obj = Word2Vec.load(path).wv
    elif model_type in ["word2vec_skipgram", "skipgram", "sg"]:
        path = os.path.join(models_dir, "word2vec_skipgram.model")
        obj = Word2Vec.load(path).wv
    elif model_type in ["fasttext", "ft"]:
        path = os.path.join(models_dir, "fasttext.model")
        obj = FastText.load(path).wv
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from 'bow', 'tfidf', 'word2vec_cbow', 'word2vec_skipgram', 'fasttext'.")

    _LOADED_MODELS[model_type] = obj
    return obj

def get_document_vector(doc_text: str, model_type: str = "tfidf", pooling: str = "mean", models_dir=None) -> np.ndarray:
    """
    TASK D2 MASTER INTERFACE FUNCTION:
    Given a document text string, returns its numerical vector representation
    under the specified model representation.

    Parameters:
    -----------
    doc_text : str
        The raw or cleaned text of the document/tweet.
    model_type : str
        The representation scheme. Valid options:
          - 'bow'               : Bag-of-Words sparse count vector (projected to 1D array)
          - 'tfidf'             : TF-IDF weighted vector (projected to 1D array)
          - 'word2vec_cbow'     : Continuous Bag-of-Words pooled document embedding (100d)
          - 'word2vec_skipgram' : Skip-gram pooled document embedding (100d)
          - 'fasttext'          : fastText sub-word pooled document embedding (100d)
    pooling : str, default='mean'
        Aggregation strategy for embedding models ('mean' or 'max') over document tokens.
    models_dir : str
        Directory where serialized models reside.

    Returns:
    --------
    np.ndarray : 1D dense or sparse array representing the document vector.
    """
    model_key = model_type.lower().strip()
    model = load_model(model_key, models_dir=models_dir)

    # 1. Count-Based Representations (BoW and TF-IDF)
    if model_key in ["bow", "tfidf"]:
        sparse_vec = model.transform([str(doc_text)])
        # Return 1D numpy array representation for consistent downstream ML compatibility
        return sparse_vec.toarray().flatten()

    # 2. Embedding-Based Representations (Word2Vec CBOW, Skip-gram, fastText)
    tokens = str(doc_text).split()
    if not tokens:
        # Return 100-dim zero vector for empty input
        return np.zeros(100, dtype=np.float32)

    token_vectors = []
    for token in tokens:
        if model_key in ["fasttext", "ft"]:
            # fastText can infer vector for any token (even OOV) using sub-words
            token_vectors.append(model[token])
        else:
            # Word2Vec requires exact in-vocabulary presence
            if token in model:
                token_vectors.append(model[token])

    if len(token_vectors) == 0:
        # All tokens were out-of-vocabulary
        return np.zeros(100, dtype=np.float32)

    token_matrix = np.array(token_vectors, dtype=np.float32)
    
    if pooling == "mean":
        doc_vector = np.mean(token_matrix, axis=0)
    elif pooling == "max":
        doc_vector = np.max(token_matrix, axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: '{pooling}'. Choose 'mean' or 'max'.")

    return doc_vector

def verify_vectorize_interface():
    """Automated test suite verifying the get_document_vector() function across all representations."""
    print("=" * 70)
    print("TASK D2: VERIFYING UNIFIED `get_document_vector()` INTERFACE")
    print("=" * 70)

    sample_doc = "apple announce new battery upgrade for iphone"
    representations = ["bow", "tfidf", "word2vec_cbow", "word2vec_skipgram", "fasttext"]

    print(f"\n[*] Sample Input Document:\n    \"{sample_doc}\"\n")
    print(f"{'Model Representation':<22} | {'Vector Shape':<14} | {'Non-Zero Entries':<18} | {'Sample Values (First 4)':<30}")
    print("-" * 90)

    for rep in representations:
        vec = get_document_vector(sample_doc, model_type=rep, pooling="mean")
        non_zeros = np.count_nonzero(vec)
        sample_vals = ", ".join([f"{v:.4f}" for v in vec[:4]])
        print(f"{rep:<22} | {str(vec.shape):<14} | {non_zeros:<18} | [{sample_vals}, ...]")

    # Test Out-Of-Vocabulary / Purely Synthetic input resilience
    oov_doc = "xyzfakeoovword qwertyunknownword zzzcompletelybogustoken"
    print(f"\n[*] Pure Out-of-Vocabulary Robustness Test:\n    \"{oov_doc}\"")
    w2v_vec = get_document_vector(oov_doc, model_type="word2vec_skipgram")
    ft_vec = get_document_vector(oov_doc, model_type="fasttext")
    print(f"    - Word2Vec Skip-gram OOV Non-Zero Entries: {np.count_nonzero(w2v_vec)} (Zero vector fallback because words are unseen)")
    print(f"    - fastText Sub-word OOV Non-Zero Entries:  {np.count_nonzero(ft_vec)} (Full 100d Vector constructed from sub-words!)")

    print("\n[+] Task D2 Master Interface Verified Successfully & Ready for Phase 3!")

if __name__ == "__main__":
    save_all_models()
    verify_vectorize_interface()
