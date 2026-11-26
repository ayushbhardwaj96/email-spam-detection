# 🛡️ Intelligent Email & SMS Spam Detection System

An automated end-to-end Machine Learning and Natural Language Processing (NLP) system designed to accurately classify emails and SMS messages as **Spam** or **Legitimate (Ham)**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

Spam emails and SMS messages pose security risks, financial threats (phishing, lottery scams), and cause communication clutter. This project implements a complete text classification pipeline featuring:

- **Text Preprocessing:** Cleans, tokenizes, removes stopwords, and lemmatizes raw text.
- **Feature Extraction:** Transforms unstructured text into numerical TF-IDF vectors (3,000 top features).
- **Multi-Algorithm Classification:** Evaluates and compares four core algorithms:
  1. **Naive Bayes** (BernoulliNB & MultinomialNB)
  2. **Support Vector Machine (SVM)**
  3. **Logistic Regression**
  4. **Decision Tree**
- **Interactive Web Application:** Built using **Streamlit** with real-time confidence scores, side-by-side multi-model comparison, and preprocessing inspection.
- **CLI Tool:** Direct command-line interface for rapid automated batch testing.

---

## 📊 Dataset Information

- **Dataset:** [SMS Spam Collection – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Total Records:** 5,572 labeled messages (`ham` / `spam`)
- **After Deduplication:** 5,169 unique messages (4,516 legitimate Ham, 653 Spam)
- **Class Balance:** ~87.4% Ham, ~12.6% Spam

---

## 🔬 NLP & Machine Learning Pipeline

```
Raw Text Message
       │
       ▼
[ Text Preprocessing ]
  ├── 1. Lowercase Normalization
  ├── 2. Punctuation & Special Character Removal
  ├── 3. Tokenization
  ├── 4. English Stopwords Removal (NLTK)
  └── 5. Word Lemmatization (WordNetLemmatizer)
       │
       ▼
[ Feature Extraction ]
  └── TF-IDF Vectorizer (max_features=3000)
       │
       ▼
[ Classification Algorithms ]
  ├── Naive Bayes (BernoulliNB / MultinomialNB)
  ├── Support Vector Machine (Linear SVM with probability)
  ├── Logistic Regression
  └── Decision Tree Classifier
       │
       ▼
[ Prediction Output ]
  └── 🚨 SPAM  or  ✅ HAM (with Confidence Score & Probability)
```

---

## 📈 Model Performance & Benchmark

All four algorithms were trained and benchmarked on a stratified 20% test split (1,034 messages):

| Algorithm | Test Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (BernoulliNB)** | **98.36%** | **96.72%** | **90.08%** | **93.28%** | ~0.13s |
| **Naive Bayes (MultinomialNB)** | 97.68% | **100.00%** | 81.68% | 89.92% | ~0.03s |
| **Support Vector Machine (SVM)** | **98.07%** | **96.64%** | **87.79%** | **92.00%** | ~47.2s |
| **Decision Tree Classifier** | 96.32% | 85.50% | 85.50% | 85.50% | ~7.98s |
| **Logistic Regression** | 95.45% | 96.67% | 66.41% | 78.73% | ~0.30s |

> **Key Observation:** In spam filtering, **Precision** is vital to ensure important legitimate communications are never mislabeled as spam. **Multinomial Naive Bayes** achieved **100% precision** (zero false positives), while **BernoulliNB** and **SVM** delivered the highest overall accuracy (~98.1% - 98.4%).

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ayushbhardwaj96/Email-Spam-Detection.git
cd Email-Spam-Detection
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Or manually: `pip install streamlit scikit-learn pandas numpy nltk`)*

---

## 💻 How to Run the Project

### Option A: Interactive Web UI (Streamlit)
To start the web application on localhost:

```bash
streamlit run spam_detector.py
```

Open your browser at: **`http://localhost:8501`**

#### App Features:
- **Algorithm Selector:** Choose between individual models or compare all 4 algorithms side-by-side.
- **Instant Preloaded Samples:** One-click buttons to test lottery spam, promo spam, lunch plans, or job offers.
- **Pipeline Breakdown:** Expandable view displaying cleaned tokens and TF-IDF feature vector dimensions.
- **Benchmark Panel:** Live sidebar metrics scorecard comparing all models.

---

### Option B: Command Line Interface (CLI)
Run classifications directly from the terminal:

```bash
# Run the built-in verification test suite:
python run_cli.py

# Test any custom message:
python run_cli.py -m "WINNER!! As a valued customer you have won a $1000 prize. Call 08002986030 now to claim."
```

---

## 📁 Project Structure

```
Email-Spam-Detection/
│
├── spam_detector.py          # Main Streamlit web application
├── run_cli.py                # Command-line interface prediction tool
├── requirements.txt          # Python project dependencies
├── Readme.md                 # Project documentation
├── .gitignore                # Git ignore configuration
│
├── models/                   # Serialized trained models & benchmarks
│   ├── naive_bayes_bernoullinb.pkl
│   ├── naive_bayes_multinomialnb.pkl
│   ├── support_vector_machine_svm.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── vectorizer.pkl
│   └── benchmark_results.csv
│
├── Spam-detection Model/     # Dataset and exploratory data analysis
│   ├── data.txt              # UCI SMS Spam Collection dataset
│   └── Spam_Mail_Detection.ipynb # Jupyter Notebook for model development
│
└── nltk_data/                # Local NLTK corpora (stopwords, wordnet, punkt)
```

---

## 👤 Author

* **Ayush Ranjan**
* GitHub: [@ayushbhardwaj96](https://github.com/ayushbhardwaj96)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
