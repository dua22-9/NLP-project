"""
exploration.py
Task 2: Basic exploration & statistics on our corpus
Author: Person 1 (Team Lead)

Calculates the required Task 2 metrics:
- Number of documents and word length stats
- Target distribution
- Tokens vs. types (vocabulary and TTR)
- Zipf's Law distribution and long tail analysis
- 5 real examples of messy tweets to justify our cleaning choices
"""

import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sentiment_tweets.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def split_words(text):
    """Simple whitespace splitter for basic exploration."""
    return re.findall(r"\S+", str(text))


def run_exploration():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}, fetching it now...")
        from load_data import load_dataset
        df = load_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    total_docs = len(df)
    print(f"\nTask 2 Exploration Results")
    print(f"Total documents: {total_docs:,}")

    # Document length in words
    df["word_count"] = df["text"].apply(lambda t: len(split_words(t)))
    mean_len = df["word_count"].mean()
    med_len = df["word_count"].median()
    min_len = df["word_count"].min()
    max_len = df["word_count"].max()
    std_len = df["word_count"].std()

    print(f"\nDocument word lengths:")
    print(f"  Mean length:   {mean_len:.2f} words")
    print(f"  Median length: {med_len:.1f} words")
    print(f"  Shortest:      {min_len} words")
    print(f"  Longest:       {max_len} words")
    print(f"  Std deviation: {std_len:.2f} words")

    # Target class counts
    print(f"\nClass balance:")
    counts = df["sentiment"].value_counts()
    for lbl, cnt in counts.items():
        pct = (cnt / total_docs) * 100
        print(f"  {lbl:<12}: {cnt:,} tweets ({pct:.1f}%)")

    # Tokens vs Types & Type-Token Ratio
    all_tokens = [tok for text in df["text"] for tok in split_words(text)]
    num_tokens = len(all_tokens)
    num_types = len(set(all_tokens))
    ttr = num_types / num_tokens if num_tokens > 0 else 0

    print(f"\nLexical statistics (raw text):")
    print(f"  Total tokens:     {num_tokens:,}")
    print(f"  Unique types:     {num_types:,}")
    print(f"  Type-Token Ratio: {ttr:.4f}")

    # Zipf's Law analysis
    freqs = Counter(all_tokens)
    sorted_freqs = sorted(freqs.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    frequencies = np.array(sorted_freqs)

    # Plot Zipf curve
    plt.figure(figsize=(7, 4.5))
    plt.loglog(ranks, frequencies, marker=".", linestyle="none", color="#1f77b4", alpha=0.5, label="Observed tokens")
    plt.loglog(ranks, frequencies[0] / ranks, linestyle="--", color="#d62728", label="Theoretical Zipf (s=1.0)")
    plt.title("Zipf's Law: Token Rank vs. Frequency", fontsize=11, fontweight="bold")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    zipf_file = os.path.join(FIGURES_DIR, "zipf_distribution.png")
    plt.savefig(zipf_file, dpi=300)
    plt.close()

    # Plot sentiment class distribution
    plt.figure(figsize=(6, 3.5))
    counts.plot(kind="bar", color=["#4a90e2", "#95a5a6", "#e74c3c", "#2ecc71"], edgecolor="black")
    plt.title("Sentiment Target Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Tweet Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    class_file = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(class_file, dpi=300)
    plt.close()

    # Plot document word length histogram
    plt.figure(figsize=(6.5, 3.5))
    plt.hist(df["word_count"], bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    plt.axvline(mean_len, color="red", linestyle="--", label=f"Mean ({mean_len:.1f} words)")
    plt.title("Tweet Word Length Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    len_file = os.path.join(FIGURES_DIR, "doc_length_distribution.png")
    plt.savefig(len_file, dpi=300)
    plt.close()

    hapaxes = sum(1 for f in sorted_freqs if f == 1)
    hapax_pct = (hapaxes / num_types) * 100
    print(f"\nZipf Commentary:")
    print(f"  {hapaxes:,} words appear only once ({hapax_pct:.1f}% of our vocabulary).")
    print(f"  This heavy long tail is caused by raw URLs, usernames (@Apple), and typos.")
    print(f"  This shows why our cleaning pipeline in Task 4 is so important to reduce sparsity.")

    # 5 Real examples of messy text
    print(f"\n5 Real messy tweets from the dataset:")
    sample_indices = [3, 7, 12, 21, 35]
    messy_records = []
    for idx in sample_indices:
        row = df.iloc[idx]
        tweet_text = row["text"]
        label = row["sentiment"]

        issues = []
        if "&" in tweet_text:
            issues.append("HTML entities (&amp;)")
        if "http" in tweet_text:
            issues.append("Raw URL (t.co link)")
        if "@" in tweet_text:
            issues.append("User mention (@username)")
        if "#" in tweet_text:
            issues.append("Hashtag (#tag)")
        if re.search(r"[?!]{2,}", tweet_text):
            issues.append("Repeated punctuation")
        if tweet_text.isupper():
            issues.append("ALL CAPS")

        print(f"\n  [Example #{idx} - Class: {label}]")
        print(f"  Raw:    {repr(tweet_text)}")
        print(f"  Issues: {', '.join(issues) if issues else 'Mixed casing/punctuation'}")
        messy_records.append({"index": idx, "label": label, "text": tweet_text, "issues": ", ".join(issues)})

    return {
        "total_docs": total_docs,
        "mean_length": mean_len,
        "class_counts": counts.to_dict(),
        "tokens": num_tokens,
        "types": num_types,
        "ttr": ttr,
        "examples": messy_records
    }


if __name__ == "__main__":
    run_exploration()
