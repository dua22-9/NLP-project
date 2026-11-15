"""
Part A: Count-Based Representations (Tasks A1, A2, A3)
======================================================
Covers:
  - Task A1: Bag-of-Words (CountVectorizer) matrix, shape, and sparsity calculation.
  - Task A2: TF-IDF (TfidfVectorizer) matrix, high-weight vs low-weight analysis.
  - Task A3: N-grams (unigrams, bigrams, trigrams, 1-2, 1-3), vocabulary growth & sparsity trade-off.
  - Visualization: Sparsity comparison, TF-IDF top weights, and N-gram growth curve.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def compute_sparsity(matrix):
    """Calculate the percentage of zero entries in a scipy sparse matrix."""
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero = matrix.nnz
    sparsity = (1.0 - (non_zero / total_elements)) * 100.0
    return sparsity, non_zero, total_elements

def run_count_based_analysis(data_path="assignment_2/data/cleaned_corpus.csv", figures_dir="assignment_2/figures"):
    """Execute all Task A analyses and save visual artifacts."""
    if not os.path.exists("assignment_2") and os.path.exists("figures"):
        figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("PART A: COUNT-BASED REPRESENTATIONS")
    print("=" * 70)
    
    # 1. Load Cleaned Corpus from Phase 1
    if not os.path.exists(data_path):
        # Fallback to relative path if run from root or src
        if os.path.exists("assignment_2/data/cleaned_corpus.csv"):
            data_path = "assignment_2/data/cleaned_corpus.csv"
        elif os.path.exists("../data/cleaned_corpus.csv"):
            data_path = "../data/cleaned_corpus.csv"
        elif os.path.exists("person_2/data/cleaned_corpus.csv"):
            data_path = "person_2/data/cleaned_corpus.csv"

    print(f"\n[+] Loading cleaned corpus from: {data_path}")
    df = pd.read_csv(data_path)
    docs = df['cleaned_text'].fillna('').astype(str).tolist()
    n_docs = len(docs)
    print(f"    Loaded {n_docs} documents successfully.")

    # -------------------------------------------------------------------------
    # Task A1: Bag of Words (BoW)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("TASK A1: BAG-OF-WORDS (COUNT VECTORIZER)")
    print("-" * 50)
    bow_vec = CountVectorizer()
    bow_matrix = bow_vec.fit_transform(docs)
    bow_vocab = bow_vec.get_feature_names_out()
    bow_sparsity, bow_nnz, bow_total = compute_sparsity(bow_matrix)

    print(f"[*] BoW Matrix Shape:     {bow_matrix.shape[0]} documents x {bow_matrix.shape[1]} terms")
    print(f"[*] Total Elements:       {bow_total:,}")
    print(f"[*] Non-Zero Elements:    {bow_nnz:,}")
    print(f"[*] BoW Matrix Sparsity:  {bow_sparsity:.4f}%")
    print(f"[*] Vocabulary Size:      {len(bow_vocab):,} unique words")

    # -------------------------------------------------------------------------
    # Task A2: TF-IDF Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("TASK A2: TF-IDF MATRIX & HIGH/LOW WEIGHT ANALYSIS")
    print("-" * 50)
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(docs)
    tfidf_vocab = tfidf_vec.get_feature_names_out()
    tfidf_sparsity, tfidf_nnz, _ = compute_sparsity(tfidf_matrix)

    print(f"[*] TF-IDF Matrix Shape:    {tfidf_matrix.shape[0]} documents x {tfidf_matrix.shape[1]} terms")
    print(f"[*] TF-IDF Matrix Sparsity: {tfidf_sparsity:.4f}%")

    # Document-level TF-IDF inspection
    # Select document 1 (informative tech tweet)
    sample_doc_idx = 1
    sample_text = docs[sample_doc_idx]
    sample_vector = tfidf_matrix[sample_doc_idx].toarray().flatten()
    
    # Get top 5 highest weighted words in this document
    non_zero_indices = sample_vector.nonzero()[0]
    doc_word_weights = [(tfidf_vocab[idx], sample_vector[idx]) for idx in non_zero_indices]
    doc_word_weights_sorted = sorted(doc_word_weights, key=lambda x: x[1], reverse=True)

    print(f"\n[*] Concrete Document Example (Document Index {sample_doc_idx}):")
    print(f"    Raw Cleaned Text: \"{sample_text}\"")
    print("    Weights for words present in this document:")
    for word, weight in doc_word_weights_sorted:
        df_count = np.sum(bow_matrix[:, bow_vec.vocabulary_[word]].toarray() > 0)
        idf_val = tfidf_vec.idf_[tfidf_vec.vocabulary_[word]]
        print(f"      - '{word}': TF-IDF={weight:.4f} | IDF={idf_val:.4f} | Doc Frequency={df_count}/{n_docs}")

    # Corpus-wide Global IDF Analysis: Highest vs Lowest IDF words
    idfs = tfidf_vec.idf_
    sorted_idf_indices = np.argsort(idfs)
    
    lowest_idf_words = [(tfidf_vocab[i], idfs[i], np.sum(bow_matrix[:, bow_vec.vocabulary_[tfidf_vocab[i]]].toarray() > 0)) 
                        for i in sorted_idf_indices[:8]]
    highest_idf_words = [(tfidf_vocab[i], idfs[i], np.sum(bow_matrix[:, bow_vec.vocabulary_[tfidf_vocab[i]]].toarray() > 0)) 
                         for i in sorted_idf_indices[-8:]]

    print("\n[*] Lowest IDF Words (Pervasive across corpus -> Down-weighted):")
    for w, idf_val, df_cnt in lowest_idf_words:
        print(f"      - '{w}': IDF={idf_val:.4f} (appears in {df_cnt} documents)")

    print("\n[*] Highest IDF Words (Rare / Specialized -> High Discriminative Power):")
    for w, idf_val, df_cnt in highest_idf_words:
        print(f"      - '{w}': IDF={idf_val:.4f} (appears in {df_cnt} document)")

    # -------------------------------------------------------------------------
    # Task A3: N-grams & Vocabulary Growth
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("TASK A3: N-GRAMS & VOCABULARY GROWTH ANALYSIS")
    print("-" * 50)
    
    ngram_configs = [
        ("Unigrams (1,1)", (1, 1)),
        ("Bigrams (2,2)", (2, 2)),
        ("Trigrams (3,3)", (3, 3)),
        ("Unigrams + Bigrams (1,2)", (1, 2)),
        ("Unigrams + Bigrams + Trigrams (1,3)", (1, 3))
    ]

    ngram_results = []
    for label, n_range in ngram_configs:
        vec = CountVectorizer(ngram_range=n_range)
        mat = vec.fit_transform(docs)
        sp, nnz, tot = compute_sparsity(mat)
        vocab_len = len(vec.vocabulary_)
        avg_tokens_per_doc = nnz / n_docs
        ngram_results.append({
            "N-Gram Scheme": label,
            "Vocabulary Size": vocab_len,
            "Non-Zero Count": nnz,
            "Sparsity (%)": sp,
            "Avg Non-Zero / Doc": avg_tokens_per_doc
        })
        print(f"[*] {label:35s} | Vocab: {vocab_len:8,d} | Sparsity: {sp:.4f}% | Avg Non-Zero/Doc: {avg_tokens_per_doc:.2f}")

    df_ngram = pd.DataFrame(ngram_results)

    # -------------------------------------------------------------------------
    # Save Figures for Report
    # -------------------------------------------------------------------------
    print("\n[+] Generating visual plots for Part A...")
    
    # Figure 1: N-gram Vocabulary Growth & Sparsity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Vocab growth
    schemes = [r["N-Gram Scheme"].split(" ")[0] for r in ngram_results]
    vocabs = [r["Vocabulary Size"] for r in ngram_results]
    bars = ax1.bar(schemes, vocabs, color=['#2b5c8f', '#3b82f6', '#60a5fa', '#10b981', '#059669'], edgecolor='black')
    ax1.set_title("Vocabulary Explosion Across N-Gram Schemes", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Unique Vocabulary (Feature Count)", fontsize=11)
    ax1.set_xticks(range(len(schemes)))
    ax1.set_xticklabels(schemes, rotation=25, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:,}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Sparsity curve
    sparsities = [r["Sparsity (%)"] for r in ngram_results]
    ax2.plot(schemes, sparsities, marker='o', color='#dc2626', linewidth=2.5, markersize=8)
    ax2.set_title("Matrix Sparsity Across N-Gram Configurations", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Sparsity (%)", fontsize=11)
    ax2.set_xticks(range(len(schemes)))
    ax2.set_xticklabels(schemes, rotation=25, ha='right')
    ax2.set_ylim(99.0, 100.0)
    ax2.grid(True, linestyle='--', alpha=0.5)
    for i, txt in enumerate(sparsities):
        ax2.annotate(f"{txt:.4f}%", (schemes[i], sparsities[i]), textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=9, fontweight='bold', color='#991b1b')

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "ngram_vocab_growth.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"    Saved: {fig1_path}")

    # Figure 2: TF-IDF High vs Low Weight Distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_words = [w[0] for w in lowest_idf_words[:5]] + [w[0] for w in highest_idf_words[:5]]
    plot_idfs = [w[1] for w in lowest_idf_words[:5]] + [w[1] for w in highest_idf_words[:5]]
    colors = ['#f87171'] * 5 + ['#34d399'] * 5
    
    y_pos = np.arange(len(plot_words))
    ax.barh(y_pos, plot_idfs, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_words, fontsize=10, fontweight='medium')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Inverse Document Frequency (IDF) Weight", fontsize=11, fontweight='bold')
    ax.set_title("TF-IDF Contrast: Common Global Words (Red) vs. Discriminative Terms (Green)", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(plot_idfs):
        ax.text(v + 0.1, i, f"{v:.2f}", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "tfidf_high_low_weights.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"    Saved: {fig2_path}")

    print("\n[+] Part A Completed Successfully!")
    return {
        "bow_shape": bow_matrix.shape,
        "bow_sparsity": bow_sparsity,
        "tfidf_shape": tfidf_matrix.shape,
        "tfidf_sparsity": tfidf_sparsity,
        "ngram_df": df_ngram
    }

if __name__ == "__main__":
    run_count_based_analysis()
