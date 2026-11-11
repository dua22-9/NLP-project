import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tokenization import compare_tokenizers, test_regex_patterns
from cleaning_pipeline import compare_stemming_and_lemmatization
from save_cleaned_data import run_and_save


def main():
    print("Running Person 2 Tasks (Tokenization, Cleaning Pipeline & Output)")

    print("\nTask 3: Running Tokenizer Benchmarking and Regex Patterns...")
    compare_tokenizers()
    test_regex_patterns()

    print("\nTask 4: Comparing Stemming vs. Lemmatization...")
    compare_stemming_and_lemmatization()

    print("\nTask 5: Running Cleaning Pipeline on Full Dataset & Exporting Cleaned Corpus...")
    run_and_save()

    print("\nPerson 2 tasks finished successfully.")
    print("Cleaned corpus saved in person_2/data/cleaned_corpus.csv")
    print("Reduction chart saved in person_2/figures/vocab_reduction.png")
    print("Notebook ready at person_2/notebooks/02_tokenization_and_cleaning.ipynb")


if __name__ == "__main__":
    main()
