"""
Assignment 2 Master Runner
===========================
Executes the full Phase 2 Natural Language Processing pipeline:
  - Part A: Count-Based Representations (Tasks A1, A2, A3: BoW, TF-IDF, N-Grams)
  - Task A4: Hand-Worked Numerical Calculations (TF-IDF & Bigram derivations + library match)
  - Part B: Learned Embeddings (Tasks B1–B4: Word2Vec CBOW, Skip-gram, fastText & OOV test)
  - Part C: Embedding Probing & Semantic Comparisons (Tasks C1–C4: Top-N, Cosine, Analogy, Synthesis)
  - Part D: Packaging for Phase 3 (Tasks D1, D2: Model serialization & unified get_document_vector() interface)
"""

import sys
import os
import time

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from count_based import run_count_based_analysis
from hand_worked_numerical import run_hand_worked_numerical
from embeddings import prepare_tokenized_corpus, train_embeddings
from probing_comparison import run_probing_and_comparison
from vectorize import save_all_models, verify_vectorize_interface

def main():
    start_time = time.time()
    print("=" * 80)
    print("ASSIGNMENT 2 — WORD REPRESENTATIONS (FULL PIPELINE)")
    print("=" * 80)

    # 1. Part A: Count-Based Representations
    print("\n>>> [1/5] EXECUTING PART A: COUNT-BASED ANALYSIS (BoW, TF-IDF, N-GRAMS)...")
    run_count_based_analysis(
        data_path=os.path.join(os.path.dirname(__file__), "data", "cleaned_corpus.csv"),
        figures_dir=os.path.join(os.path.dirname(__file__), "figures")
    )

    # 2. Task A4: Hand-Worked Numerical Derivation & Verification
    print("\n>>> [2/5] EXECUTING TASK A4: HAND-WORKED NUMERICAL VERIFICATION...")
    run_hand_worked_numerical()

    # 3. Part B: Learned Embeddings (CBOW, Skip-gram, fastText)
    print("\n>>> [3/5] EXECUTING PART B: TRAINING LEARNED EMBEDDINGS...")
    sentences = prepare_tokenized_corpus(
        data_path=os.path.join(os.path.dirname(__file__), "data", "cleaned_corpus.csv")
    )
    train_embeddings(
        sentences,
        models_dir=os.path.join(os.path.dirname(__file__), "models"),
        figures_dir=os.path.join(os.path.dirname(__file__), "figures")
    )

    # 4. Part C: Probing, Similarities, Analogies
    print("\n>>> [4/5] EXECUTING PART C: EMBEDDING PROBING & SEMANTIC COMPARISONS...")
    run_probing_and_comparison(
        models_dir=os.path.join(os.path.dirname(__file__), "models"),
        figures_dir=os.path.join(os.path.dirname(__file__), "figures")
    )

    # 5. Part D: Packaging & Vectorize Interface Test
    print("\n>>> [5/5] EXECUTING PART D: PACKAGING MODELS & TESTING VECTORIZE API...")
    save_all_models(
        data_path=os.path.join(os.path.dirname(__file__), "data", "cleaned_corpus.csv"),
        models_dir=os.path.join(os.path.dirname(__file__), "models")
    )
    verify_vectorize_interface()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ASSIGNMENT 2 FULL PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("Models serialized in:  assignment_2/models/")
    print("Figures saved in:      assignment_2/figures/")
    print("=" * 80)

if __name__ == "__main__":
    main()
