"""
save_cleaned_data.py
Task 5: Running our cleaning pipeline on the full dataset & exporting it
Author: Person 2 (Group Member 1)

This script applies clean_text() to all 5,113 Kaggle tweets,
measures how much our vocabulary shrank after cleaning,
saves the reduction chart to figures/vocab_reduction.png,
and exports the cleaned data to data/cleaned_corpus.csv for Phase 2!
"""

import os
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
from cleaning_pipeline import clean_text

RAW_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sentiment_tweets.csv")
CLEANED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_corpus.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
KAGGLE_SOURCE_URL = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"


def ensure_raw_data():
    """Ensures raw dataset is available; fetches from Kaggle benchmark mirror if missing."""
    if not os.path.exists(RAW_CSV):
        os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
        print(f"Raw data not found locally. Fetching Kaggle dataset from: {KAGGLE_SOURCE_URL}")
        try:
            req = urllib.request.Request(KAGGLE_SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                raw_df = pd.read_csv(response)

            df = raw_df.rename(columns={
                "TweetId": "tweet_id",
                "TweetText": "text",
                "Sentiment": "sentiment",
                "Topic": "topic",
                "TweetDate": "date"
            })
            df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
            df.to_csv(RAW_CSV, index=False, encoding="utf-8")
            print(f"Downloaded and saved {len(df):,} tweets to {RAW_CSV}")
        except Exception as e:
            raise RuntimeError(f"Could not load raw data: {e}")


def run_and_save():
    ensure_raw_data()
    df = pd.read_csv(RAW_CSV)
    print(f"\nTask 5: Processing Full Dataset ({len(df):,} tweets)")

    # Raw vocabulary metrics
    raw_tokens = [tok for text in df["text"] for tok in str(text).split()]
    raw_token_count = len(raw_tokens)
    raw_vocab_count = len(set(raw_tokens))

    # Apply cleaning pipeline
    print("Applying clean_text() pipeline across all tweets...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Cleaned vocabulary metrics
    clean_tokens = [tok for text in df["cleaned_text"] for tok in str(text).split()]
    clean_token_count = len(clean_tokens)
    clean_vocab_count = len(set(clean_tokens))

    # Calculate percentage drops
    token_drop = ((raw_token_count - clean_token_count) / raw_token_count) * 100
    vocab_drop = ((raw_vocab_count - clean_vocab_count) / raw_vocab_count) * 100

    print("\nVocabulary Reduction Summary:")
    print(f"  Total tokens:     {raw_token_count:,} -> {clean_token_count:,} (-{token_drop:.2f}%)")
    print(f"  Unique vocabulary:{raw_vocab_count:,} -> {clean_vocab_count:,} (-{vocab_drop:.2f}%)")

    # Plot the before vs after comparison
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Total tokens bar chart
    axes[0].bar(["Before", "After"], [raw_token_count, clean_token_count], color=["#e74c3c", "#2ecc71"], edgecolor="black")
    axes[0].set_title("Total Running Tokens", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Token Count")
    axes[0].text(0, raw_token_count * 0.5, f"{raw_token_count:,}", ha="center", color="white", fontweight="bold")
    axes[0].text(1, clean_token_count * 0.5, f"{clean_token_count:,}\n(-{token_drop:.1f}%)", ha="center", color="white", fontweight="bold")

    # Vocab types bar chart
    axes[1].bar(["Before", "After"], [raw_vocab_count, clean_vocab_count], color=["#9b59b6", "#3498db"], edgecolor="black")
    axes[1].set_title("Unique Vocabulary (Types)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Type Count")
    axes[1].text(0, raw_vocab_count * 0.5, f"{raw_vocab_count:,}", ha="center", color="white", fontweight="bold")
    axes[1].text(1, clean_vocab_count * 0.5, f"{clean_vocab_count:,}\n(-{vocab_drop:.1f}%)", ha="center", color="white", fontweight="bold")

    plt.suptitle("Vocabulary & Token Reduction (Phase 1 Cleaning)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart_file = os.path.join(FIGURES_DIR, "vocab_reduction.png")
    plt.savefig(chart_file, dpi=300)
    plt.close()

    # Save cleaned corpus
    clean_df = df[["tweet_id", "text", "cleaned_text", "sentiment"]].rename(columns={"text": "raw_text"})
    clean_df.to_csv(CLEANED_CSV, index=False, encoding="utf-8")
    print(f"\nCleaned corpus exported to: {CLEANED_CSV}")
    print(f"Chart saved to: {chart_file}")

    print("\nFirst 3 Cleaned Examples:")
    for _, row in clean_df.head(3).iterrows():
        print(f"  [{row['sentiment'].upper()}] Raw:     {row['raw_text'].strip()[:65]}...")
        print(f"  [{row['sentiment'].upper()}] Cleaned: {row['cleaned_text']}\n")

    return clean_df


if __name__ == "__main__":
    run_and_save()
