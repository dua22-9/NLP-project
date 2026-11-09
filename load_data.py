import os
import urllib.request
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_FILE = os.path.join(RAW_DIR, "sentiment_tweets.csv")
KAGGLE_SOURCE_URL = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"


def load_dataset():
    """Loads the dataset from local cache if it exists, otherwise downloads it."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_FILE):
        print(f"Reading cached dataset from: {RAW_FILE}")
        df = pd.read_csv(RAW_FILE)
    else:
        print(f"Downloading Kaggle Twitter corpus from: {KAGGLE_SOURCE_URL}")
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
            df.to_csv(RAW_FILE, index=False, encoding="utf-8")
            print(f"Saved {len(df):,} tweets to {RAW_FILE}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch dataset: {e}")

    print(f"Total documents: {len(df):,}")
    print("Columns available:", list(df.columns))
    print("\nClass breakdown:")
    for label, count in df["sentiment"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  {label:<12}: {count:,} ({pct:.1f}%)")

    return df


if __name__ == "__main__":
    load_dataset()
