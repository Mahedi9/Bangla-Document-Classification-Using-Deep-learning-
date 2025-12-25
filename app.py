import streamlit as st
import numpy as np
import re
import joblib
import torch
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bnltk.stemmer import BanglaStemmer

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bangla Document Classification Using Deep learning ",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
# 📄 Bangla Document Classification Using Deep learning  
### Enhancing Bangla Document Classification Using a Hybrid Ensemble of Bangla-BERT and Bi-LSTM Models  
**(IDAA 2025 – Published Research)**
""")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_bilstm():
    model = load_model("bilstm_model.keras")
    tokenizer = joblib.load("bilstm_tokenizer.pkl")
    le = joblib.load("bilstm_label_encoder.pkl")
    return model, tokenizer, le

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("bangla_bert_model")
    model = AutoModelForSequenceClassification.from_pretrained("bangla_bert_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

bilstm_model, bilstm_tokenizer, label_encoder = load_bilstm()
bert_tokenizer, bert_model, device = load_bert()

# ===============================
# PREPROCESSING (UNCHANGED)
# ===============================
def clean_article(text):
    text = re.sub(r'[^\u0980-\u09FF\s]', '', str(text))
    text = re.sub(r'\n|\t|\r|\xa0', ' ', text)
    text = re.sub(r'[^ঀ-৿a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

bn_stemmer = BanglaStemmer()

bangla_stopwords = [
    'অতএব','অথচ','অথবা','অনুযায়ী','অনেক','অনেকে','অনেকেই','অন্তত','অন্য','অবধি',
    'অবশ্য','অর্থাত','আগে','আগেই','আছে','আজ','আপনি','আমরা','আমাকে','আমাদের','আমার',
    'আমি','আর','আরও','ই','ইত্যাদি','উচিত','উপর','এ','এই','এক','একটি','এখন','এটা',
    'এবং','এমন','এর','ও','কখনও','কত','কবে','কয়েক','করছে','করতে','করবে','করা',
    'করে','করেছে','করেন','কিছু','কিন্তু','কে','কেউ','কেন','কোন','কোনও','খুব',
    'ছিল','ছিলেন','জন','জন্য','জানতে','জানা','ঠিক','তখন','তবে','তা','তাকে',
    'তাদের','তার','তারা','তিনি','তুমি','থাকবে','থেকে','দিতে','দিয়ে','দেন',
    'দুই','দুটি','দেওয়া','দেখতে','দেখা','দেখে','নয়','না','নিয়ে','নেই','পরে',
    'পর','প্রথম','প্রায়','ফলে','বলতে','বলে','বলেন','বেশি','মতো','মধ্যে',
    'মনে','যখন','যদি','যা','যাকে','যাতে','যাদের','যায়','যার','যারা','যিনি',
    'যে','রয়েছে','লক্ষ','শুধু','সব','সঙ্গে','সহ','সে','সেই','সেখানে',
    'হতে','হয়','হবে','হয়ে','হল','হলে'
]

def remove_stopwords(text):
    words = text.split()
    return " ".join([bn_stemmer.stem(w) for w in words if w not in bangla_stopwords])

# ===============================
# PREDICTION FUNCTIONS
# ===============================
def predict_bilstm(text):
    t = remove_stopwords(clean_article(text))
    seq = bilstm_tokenizer.texts_to_sequences([t])
    pad = pad_sequences(seq, maxlen=400, padding='post')
    return bilstm_model.predict(pad)[0]

def predict_bert(text):
    t = clean_article(text)
    inputs = bert_tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

# ===============================
# CONFIDENCE THRESHOLD LOGIC
# ===============================
CONF_THRESHOLD = 0.15

def final_decision(probs, classes):
    idx_sorted = np.argsort(probs)
    top1, top2 = idx_sorted[-1], idx_sorted[-2]

    if probs[top1] - probs[top2] < CONF_THRESHOLD:
        return f"Ambiguous ({classes[top1]} / {classes[top2]})", False
    else:
        return classes[top1], True

# ===============================
# UI INPUT
# ===============================
st.subheader("✍️ Input Bangla Article")
text = st.text_area("Paste Bangla news text:", height=220)

if st.button("🔍 Predict"):
    if len(text.split()) < 20:
        st.warning("⚠️ Please provide at least 20 words.")
    else:
        with st.spinner("Running models..."):
            bl_probs = predict_bilstm(text)
            br_probs = predict_bert(text)
            en_probs = 0.5 * bl_probs + 0.5 * br_probs

            classes = label_encoder.classes_

            final_label, confident = final_decision(en_probs, classes)

        # ===============================
        # FINAL RESULT SECTION
        # ===============================
        st.markdown("## 🏆 FINAL RESULT")
        if confident:
            st.success(f"### ✅ {final_label}")
        else:
            st.warning(f"### ⚠️ {final_label}")
            st.info( "This article contains overlapping semantic themes. " "The system applied a confidence-threshold-based decision ""to avoid forced misclassification.")


        # ===============================
        # MODEL-WISE RESULTS
        # ===============================
        col1, col2, col3 = st.columns(3)
        col1.metric("Bi-LSTM", classes[np.argmax(bl_probs)], f"{np.max(bl_probs)*100:.2f}%")
        col2.metric("Bangla-BERT", classes[np.argmax(br_probs)], f"{np.max(br_probs)*100:.2f}%")
        col3.metric("Hybrid Ensemble", classes[np.argmax(en_probs)], f"{np.max(en_probs)*100:.2f}%")

        # ===============================
        # PROBABILITY BAR CHARTS
        # ===============================
        st.subheader("📊 Probability Distribution")

        def plot_bar(probs, title):
            fig, ax = plt.subplots()
            ax.barh(classes, probs)
            ax.set_xlim(0, 1)
            ax.set_title(title)
            st.pyplot(fig)

        c1, c2, c3 = st.columns(3)
        with c1: plot_bar(bl_probs, "Bi-LSTM")
        with c2: plot_bar(br_probs, "Bangla-BERT")
        with c3: plot_bar(en_probs, "Hybrid Ensemble")

        st.subheader("📊 Class-wise Probabilities")
        df = pd.DataFrame({
            "Class": classes,
            "Bi-LSTM": bl_probs,
            "Bangla-BERT": br_probs,
            "Hybrid Ensemble": en_probs
        })

        st.dataframe( df.style.format({
        "Bi-LSTM": "{:.4f}",
        "Bangla-BERT": "{:.4f}",
        "Hybrid Ensemble": "{:.4f}"
    })
)


# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("IDAA 2025 | Enhancing Bangla Document Classification Using a Hybrid Ensemble of Bangla-BERT and Bi-LSTM Models | Mahedi Hasan Emon et al.")
