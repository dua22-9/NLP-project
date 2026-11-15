# NLP Semester Project — Social Media Sentiment Analysis
**Course**: Introduction to Natural Language Processing  
**Team**: `Team Alpha` (`nlp-project-teamalpha`)  
**Corpus**: 5,113 Real Tweets (Kaggle / Sanders Benchmark Corpus)  

This repository contains the complete codebase, statistical analyses, models, and execution scripts for **Assignment 1 (Phase 1)** and **Assignment 2 (Phase 2)** strictly conforming to the official course guidelines.

---

## 1. Directory Structure

```text
NLP project/
├── requirements.txt                      # Project dependencies (pandas, scikit-learn, gensim, etc.)
├── README.md                             # Main repository documentation
├── .gitignore                            # Standard git exclusions
│
├── assignment_1/                         # [ASSIGNMENT 1 / PHASE 1 DELIVERABLES]
│   ├── main.py                           # One-click master runner for Assignment 1
│   ├── data/
│   │   ├── raw/sentiment_tweets.csv      # Raw Kaggle Twitter corpus (5,113 tweets)
│   │   ├── slang_dict.csv                # Domain slang translation dictionary
│   │   └── cleaned_corpus.csv            # Cleaned corpus exported from Task 5
│   ├── figures/                          # Core figures required by Assignment 1 guidelines
│   │   ├── class_distribution.png        # Task 2: 4-class target balance
│   │   ├── doc_length_distribution.png   # Task 2: Tweet word length histogram
│   │   ├── zipf_distribution.png         # Task 2: Log-log Zipf's law distribution & tail
│   │   └── vocab_reduction.png           # Task 4: Token/vocabulary shrinkage chart
│   ├── notebooks/
│   │   ├── 01_exploration.ipynb          # Tasks 1 & 2 interactive exploration
│   │   └── 02_tokenization_and_cleaning.ipynb # Tasks 3, 4 & 5 tokenization & cleaning
│   └── src/
│       ├── load_data.py                  # Task 1: Dataset acquisition & loading
│       ├── exploration.py                # Task 2: Corpus statistics & distributions
│       ├── tokenization.py               # Task 3: Tokenizer benchmarking & domain regex
│       ├── cleaning_pipeline.py          # Task 4: Regex cleaning & lemmatization
│       └── save_cleaned_data.py          # Task 5: Batch processing & corpus export
│
├── assignment_2/                         # [ASSIGNMENT 2 / PHASE 2 DELIVERABLES - TAG 'assignment-2']
│   ├── main.py                           # One-click master runner for Assignment 2
│   ├── data/
│   │   └── cleaned_corpus.csv            # Same cleaned dataset produced in Assignment 1
│   ├── figures/                          # Figures required by Assignment 2 guidelines
│   │   ├── tfidf_high_low_weights.png    # Task A2: High/Low TF-IDF discriminative contrast
│   │   ├── ngram_vocab_growth.png        # Task A3: Feature growth & sparsity trade-off
│   │   ├── fasttext_oov_comparison.png   # Task B4: fastText sub-word OOV resilience
│   │   ├── embedding_similarities_heatmap.png # Task C2: Pairwise cosine similarity matrix
│   │   ├── handwritten_worksheet_person1.png # Task A4: CamScanner worksheet (Doc d0 / 'release')
│   │   └── handwritten_worksheet_person2.png # Task A4: CamScanner worksheet (Doc d2 / 'battery')
│   ├── models/                           # Serialized models & vectorizers (Task D1)
│   │   ├── bow_vectorizer.joblib         # Bag-of-Words CountVectorizer (9,561 features)
│   │   ├── tfidf_vectorizer.joblib       # TfidfVectorizer (9,561 features)
│   │   ├── word2vec_cbow.model           # Word2Vec CBOW (sg=0, vector_size=100)
│   │   ├── word2vec_skipgram.model       # Word2Vec Skip-gram (sg=1, vector_size=100)
│   │   └── fasttext.model                # fastText (sub-word char n-grams 3–6)
│   ├── notebooks/
│   │   ├── 01_count_based_representations.ipynb # Part A: BoW, TF-IDF, N-grams, Task A4
│   │   ├── 02_learned_embeddings_and_probing.ipynb # Parts B & C: Word2Vec, fastText, Probing
│   │   └── 03_vectorize_interface_demo.ipynb       # Part D: Unified get_document_vector() demo
│   └── src/
│       ├── count_based.py                # Part A: Count-based representations
│       ├── hand_worked_numerical.py      # Task A4: Exact arithmetic derivations & verification
│       ├── embeddings.py                 # Part B: Learned embedding models & OOV demo
│       ├── probing_comparison.py         # Part C: Probing, cosine distances & analogies
│       └── vectorize.py                  # Part D: Packaging & unified vectorize API
│
└── not_required/                         # Reference materials, study guides & archive
    ├── reports/                          # Archived Word, HTML, and Markdown report copies
    │   ├── Person1_Report.docx, .html, .md
    │   ├── Person2_Report.docx, .html, .md
    │   ├── Person1_Report_Assignment2.docx, .html, .md
    │   └── Person2_Report_Assignment2.docx, .html, .md
    ├── terminal_screenshots/             # Real terminal execution screenshots
    ├── CHECKLIST_1.md & CHECKLIST_2.md   # Official rubric scoring criteria checklists
    ├── HOW_TO_RUN_1.md & HOW_TO_RUN_2.md # Terminal execution guides
    ├── PROJECT_EXPLANATION_1.md & _2.md  # Plain-English technical explanations
    ├── VIVA_PREPARATION_1.md & _2.md     # Viva oral defense question banks
    ├── HANDWRITTEN_NUMERICAL_GUIDE_*.md  # CamScanner mathematical worksheets
    ├── HANDWRITTEN_SHEET_*.png           # Visual lined-paper worksheet reference images
    └── verify_task_a4.py                 # Automated library verification script
```

---

## 2. Quick Run Instructions

### Activate Virtual Environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Assignment 1 (Phase 1):
```powershell
python assignment_1/main.py
```
* Runs Tasks 1 to 5 end-to-end.
* Generates required figures in `assignment_1/figures/`.
* Exports cleaned corpus to `assignment_1/data/cleaned_corpus.csv`.

### Assignment 2 (Phase 2):
```powershell
python assignment_2/main.py
```
* Runs Parts A, B, C, D and Task A4 end-to-end.
* Generates required figures in `assignment_2/figures/`.
* Serializes all 5 representations into `assignment_2/models/`.
* Validates unified `get_document_vector()` API on raw text and OOV inputs.

---

## 3. Git Release Tags
- Phase 1 Release: `assignment-1`
- Phase 2 Release: `assignment-2`
