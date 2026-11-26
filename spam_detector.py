import os
import pickle
import string
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure local NLTK data directory inside workspace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")
if os.path.exists(NLTK_DIR) and NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

# Ensure required NLTK resources exist
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4")
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource, download_dir=NLTK_DIR if os.path.exists(NLTK_DIR) else None, quiet=True)
        except Exception:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

lemma = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def transform_text(text):
    """Text preprocessing: lowercase, punctuation & special character removal, stopword filtering, lemmatization."""
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ''.join(ch if ch.isalpha() or ch.isspace() else ' ' for ch in text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [lemma.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Load Vectorizer
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
if not os.path.exists(vectorizer_path):
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(vectorizer_path, "rb") as f:
    tfidf = pickle.load(f)

# Load Models
available_models = {}
models_dir = os.path.join(BASE_DIR, "models")
model_files = {
    "Naive Bayes (BernoulliNB)": "naive_bayes_bernoullinb.pkl",
    "Naive Bayes (MultinomialNB)": "naive_bayes_multinomialnb.pkl",
    "Support Vector Machine (SVM)": "support_vector_machine_svm.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl"
}

for name, filename in model_files.items():
    fp = os.path.join(models_dir, filename)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            available_models[name] = pickle.load(f)

# Fallback if models directory not populated
if not available_models and os.path.exists(os.path.join(BASE_DIR, "model.pkl")):
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        available_models["Naive Bayes (Default Saved)"] = pickle.load(f)

# Page Configuration
st.set_page_config(
    page_title="Intelligent Email/SMS Spam Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Intelligent Email & SMS Spam Detection System")
st.markdown("""
An automated classification system to distinguish spam from legitimate messages using **Text Preprocessing**, **TF-IDF Feature Extraction**, and **Machine Learning Classification** trained on the **UCI SMS Spam Collection** dataset.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration & Evaluation")
algo_options = ["All Algorithms (Comparison)"] + list(available_models.keys())
selected_algo = st.sidebar.selectbox("Select Classification Algorithm:", algo_options, index=0)

# Display Benchmark Metrics in Sidebar
benchmark_csv = os.path.join(models_dir, "benchmark_results.csv")
if os.path.exists(benchmark_csv):
    st.sidebar.subheader("📊 Algorithm Benchmark (Test Set)")
    bdf = pd.read_csv(benchmark_csv)
    bdf_display = bdf[["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score"]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        bdf_display[col] = (bdf_display[col] * 100).map("{:.2f}%".format)
    st.sidebar.dataframe(bdf_display, hide_index=True)

# Manage input field state properly
if "user_message" not in st.session_state:
    st.session_state["user_message"] = ""

def set_sample_text(text):
    st.session_state["user_message"] = text

def clear_input():
    st.session_state["user_message"] = ""

# Quick sample buttons
st.subheader("✉️ Enter Email / SMS Message")
col_s1, col_s2, col_s3, col_s4, col_clr = st.columns([1, 1, 1, 1, 0.6])

col_s1.button(
    "📋 Lottery Spam",
    on_click=set_sample_text,
    args=("WINNER!! As a valued customer you have won a £900 prize reward! Call 09061701461 to claim now. Claim code KL341.",),
    use_container_width=True
)
col_s2.button(
    "📋 Promo Spam",
    on_click=set_sample_text,
    args=("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010 now!",),
    use_container_width=True
)
col_s3.button(
    "📋 Lunch Ham",
    on_click=set_sample_text,
    args=("Hey, are we still meeting for lunch at 12:30 pm today near the central library?",),
    use_container_width=True
)
col_s4.button(
    "📋 Work Offer Ham",
    on_click=set_sample_text,
    args=("Congratulation on your selection for wells fargo as technoloy program intern.",),
    use_container_width=True
)
col_clr.button(
    "🧹 Clear",
    on_click=clear_input,
    use_container_width=True
)

input_sms = st.text_area(
    "Message text to analyze:",
    key="user_message",
    height=120,
    placeholder="Type or paste any email or SMS content here..."
)

col_btn, _ = st.columns([1, 4])
predict_btn = col_btn.button("🔍 Classify Message", type="primary", use_container_width=True)

if predict_btn:
    current_text = input_sms.strip()
    if not current_text:
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocessing & Vectorization
        transformed_sms = transform_text(current_text)
        vector_input = tfidf.transform([transformed_sms]).toarray()
        nonzero_count = int(np.count_nonzero(vector_input))

        st.markdown("---")
        st.subheader("🎯 Classification Results")

        if selected_algo == "All Algorithms (Comparison)":
            # Display models cleanly in a 3-column responsive grid
            model_items = list(available_models.items())
            num_cols = 3
            for row_idx in range(0, len(model_items), num_cols):
                cols = st.columns(num_cols)
                for c_idx, (algo_name, model_obj) in enumerate(model_items[row_idx:row_idx+num_cols]):
                    with cols[c_idx]:
                        pred = model_obj.predict(vector_input)[0]
                        prob_str = ""
                        if hasattr(model_obj, "predict_proba"):
                            try:
                                proba = model_obj.predict_proba(vector_input)[0]
                                conf = proba[pred] * 100
                                prob_str = f"Confidence: **{conf:.1f}%**"
                            except Exception:
                                pass
                        
                        if pred == 1:
                            st.error(f"### 🚨 SPAM\n**{algo_name}**\n\n{prob_str}")
                        else:
                            st.success(f"### ✅ HAM (Not Spam)\n**{algo_name}**\n\n{prob_str}")
        else:
            active_model = available_models[selected_algo]
            pred = active_model.predict(vector_input)[0]
            
            c1, c2 = st.columns([1, 2])
            with c1:
                if pred == 1:
                    st.error("## 🚨 Result: SPAM")
                else:
                    st.success("## ✅ Result: HAM (Legitimate)")
            
            with c2:
                st.write(f"**Model Used:** `{selected_algo}`")
                if hasattr(active_model, "predict_proba"):
                    try:
                        proba = active_model.predict_proba(vector_input)[0]
                        st.metric("Spam Probability", f"{proba[1]*100:.2f}%")
                        st.progress(float(proba[1]))
                    except Exception:
                        pass

        # Preprocessing breakdown expander (Fixed .nnz -> nonzero_count)
        with st.expander("🔬 View Pipeline Preprocessing Details", expanded=True):
            st.markdown(f"**Original Text:**\n`{current_text}`")
            st.markdown(f"**Cleaned & Lemmatized Tokens:**\n`{transformed_sms if transformed_sms else '[No vocabulary tokens matched]'}`")
            st.markdown(f"**TF-IDF Feature Vector Size:** `{vector_input.shape[1]} features` (Non-zero vocabulary entries: `{nonzero_count}`)")
