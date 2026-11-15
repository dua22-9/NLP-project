"""
Assignment 1 Master Runner
===========================
Executes the full Phase 1 Natural Language Processing pipeline:
  - Task 1: Dataset Acquisition & Loading (Kaggle Twitter Corpus, 5,113 documents)
  - Task 2: Corpus Exploration & Statistics (Length stats, class distribution, TTR, Zipf's law)
  - Task 3: Tokenizer Comparison & Regex Demonstrations
  - Task 4: Normalization (Lemmatization vs. Stemming justification)
  - Task 5: End-to-End Cleaning Pipeline & Export of Cleaned Corpus
"""

import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load_data import load_dataset
from exploration import run_exploration
from tokenization import compare_tokenizers, test_regex_patterns
from cleaning_pipeline import compare_stemming_and_lemmatization
from save_cleaned_data import run_and_save

def main():
    print("=" * 80)
    print("ASSIGNMENT 1 — COMPLETE NATURAL LANGUAGE PROCESSING PIPELINE")
    print("=" * 80)

    print("\n>>> TASK 1: Loading raw Kaggle Twitter corpus...")
    df = load_dataset()

    print("\n>>> TASK 2: Running corpus statistics and generating plots...")
    stats = run_exploration()

    print("\n>>> TASK 3: Running Tokenizer Benchmarking and Regex Patterns...")
    compare_tokenizers()
    test_regex_patterns()

    print("\n>>> TASK 4: Comparing Stemming vs. Lemmatization...")
    compare_stemming_and_lemmatization()

    print("\n>>> TASK 5: Running Cleaning Pipeline on Full Dataset & Exporting Cleaned Corpus...")
    run_and_save()

    print("\n" + "=" * 80)
    print("ASSIGNMENT 1 PIPELINE EXECUTED SUCCESSFULLY!")
    print("Cleaned corpus exported to: assignment_1/data/cleaned_corpus.csv")
    print("Figures saved in:           assignment_1/figures/")
    print("=" * 80)

if __name__ == "__main__":
    main()
