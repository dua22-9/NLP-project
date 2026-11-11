"""
tokenization.py
Task 3: Tokenization experiments & domain regex patterns
Author: Person 2 (Group Member 1)

Here I test two tokenization methods on tricky social media sentences:
1. Basic RegexpTokenizer r'\\w+'
2. NLTK TweetTokenizer (built specifically for microblog text)

I also created 4 regular expressions for cleaning and feature extraction:
URLs, handles/hashtags, emoticons, and character elongation.
"""

import re
from nltk.tokenize import RegexpTokenizer, TweetTokenizer


# Our 4 domain regular expressions
REGEX_PATTERNS = {
    "url_cleaner": {
        "pattern": r"https?://\S+|www\.\S+",
        "purpose": "Match and remove web URLs (http, https, www)",
        "explanation": (
            "- 'https?://' matches both http:// and https://\n"
            "- '\\S+' matches non-space characters in the domain and query path\n"
            "- '|www\\.\\S+' matches links starting with www. without http prefix"
        )
    },
    "handles_and_hashtags": {
        "pattern": r"(@[A-Za-z0-9_]{1,15}|#[A-Za-z0-9_]+)",
        "purpose": "Extract user mentions (@username) and hashtags (#topic)",
        "explanation": (
            "- '@[A-Za-z0-9_]{1,15}' matches Twitter usernames (1-15 valid chars)\n"
            "- '|' is an OR operator\n"
            "- '#[A-Za-z0-9_]+' matches hashtags with the leading '#' symbol"
        )
    },
    "emoticons": {
        "pattern": r"(?:<3|</3|[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\])",
        "purpose": "Capture ASCII hearts and emoticons to preserve emotional polarity",
        "explanation": (
            "- '(?:<3|</3)' captures heart and broken heart emoticons\n"
            "- '[:;=8]' matches eye characters\n"
            "- '[\\-o\\*\\']?' matches optional nose/tear markers\n"
            "- '[\\)\\]\\(\\[dDpP/...' matches smiles, frowns, laughing mouths"
        )
    },
    "character_elongation": {
        "pattern": r"(.)\1{2,}",
        "purpose": "Match letters repeated 3 or more times (e.g., 'soooo', 'yaaaay')",
        "explanation": (
            "- '(.)' captures any character into group 1\n"
            "- '\\1{2,}' finds 2 or more subsequent repeats of that exact character"
        )
    }
}


def compare_tokenizers():
    """Compares the two tokenizers on 3 tricky social media sentences."""
    test_sentences = [
        "OMG @sarah_j the new update is literally AMAZING!! <3 check out https://t.co/x891JkA72 #loveit #tech",
        "Why can't companies just be 100% transparent?! It's $49.99/mo -- completely unfair!! >:(",
        "TBH I was soooo skeptical at first &amp; almost quit, but 10/10 would recommend! :)"
    ]

    regex_tok = RegexpTokenizer(r"\w+")
    tweet_tok = TweetTokenizer(preserve_case=True, strip_handles=False, reduce_len=True)

    print("\nTask 3: Tokenizer Comparison on Social Media Sentences")

    for i, sent in enumerate(test_sentences, 1):
        toks_regex = regex_tok.tokenize(sent)
        toks_tweet = tweet_tok.tokenize(sent)

        print(f"\nTest Sentence {i}:")
        print(f"  Input: {sent}")
        print(f"  Method A (RegexpTokenizer r'\\w+'):")
        print(f"    Count:  {len(toks_regex)} tokens")
        print(f"    Tokens: {toks_regex}")
        print(f"  Method B (NLTK TweetTokenizer):")
        print(f"    Count:  {len(toks_tweet)} tokens")
        print(f"    Tokens: {toks_tweet}")

        if i == 1:
            print("  Analysis: RegexpTokenizer broke '<3' into an orphan number '3', shredded the URL, and stripped @ and # symbols. TweetTokenizer kept them all intact!")
        elif i == 2:
            print("  Analysis: RegexpTokenizer split 'can\\'t' into 'can' and 't', lost '$', '%', and deleted the angry emoticon '>:( '. TweetTokenizer kept them all.")
        elif i == 3:
            print("  Analysis: RegexpTokenizer turned '&amp;' into 'amp'. TweetTokenizer kept '&' and reduced 'soooo' to 'sooo'.")


def test_regex_patterns():
    """Demonstrates our 4 regular expressions on sample text."""
    print("\nDemonstration of our 4 Regular Expressions:")

    sample_strings = [
        "Check this out: https://news.ycombinator.com/item?id=892 and www.google.com for info!",
        "Big thanks to @alex_tech and @dev_lead for shipping the update! #innovation #NLP2026",
        "Feeling super blessed today <3 <3 :) while yesterday was terrible :'( >:(",
        "I am soooo exciteddddd for the big party tonightttttt!!!!"
    ]

    for name, info in REGEX_PATTERNS.items():
        print(f"\nPattern: {name.upper()}")
        print(f"  Regex:       {info['pattern']}")
        print(f"  Purpose:     {info['purpose']}")
        print(f"  Explanation: {info['explanation']}")

        matches = []
        for s in sample_strings:
            found = re.findall(info["pattern"], s)
            if found:
                matches.extend(found)
        print(f"  Matched:     {matches}")


if __name__ == "__main__":
    compare_tokenizers()
    test_regex_patterns()
