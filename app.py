import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="wide")

SPAM, NOT_SPAM = 0, 1   # tumhare dataset ke hisaab se

@st.cache_resource
def load_files():
    return joblib.load("spam_model.pkl"), joblib.load("transform.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("spam_data_csv.csv")

model, cv = load_files()
df = load_data()

@st.cache_data
def compute_stats():
    X = cv.transform(df["text"])
    y = df["label"]
    pred = model.predict(X)

    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, pos_label=SPAM),
        "Recall": recall_score(y, pred, pos_label=SPAM),
        "F1 Score": f1_score(y, pred, pos_label=SPAM),
    }
    cm = confusion_matrix(y, pred, labels=[SPAM, NOT_SPAM])

    words = cv.get_feature_names_out()
    top = {}
    for name, lab in [("Spam", SPAM), ("Not Spam", NOT_SPAM)]:
        counts = np.asarray(X[(y == lab).values].sum(axis=0)).ravel()
        top[name] = pd.Series(counts, index=words).nlargest(10)
    return metrics, cm, top

st.title("📩 Spam Message Detector")
tab1, tab2 = st.tabs(["🔍 Detector", "📊 Dashboard"])

# ---------------- TAB 1: DETECTOR ----------------
with tab1:
    st.write("Message paste karo, pata chal jayega ki wo spam hai ya nahi.")

    c1, c2 = st.columns(2)
    if c1.button("Spam example"):
        st.session_state["msg"] = ("Dear customer, your KYC is expired. "
                                   "Click www.prize-today.biz to update or your account will be closed.")
    if c2.button("Not spam example"):
        st.session_state["msg"] = "Hi, the budget meeting starts at 10 am Monday. See you there."

    msg = st.text_area("Apna message likho:", key="msg", height=150)

    if st.button("Check Karo", type="primary"):
        if not msg.strip():
            st.warning("Pehle kuch message likho.")
        else:
            vec = cv.transform([msg])
            pred = model.predict(vec)[0]
            proba = dict(zip(model.classes_, model.predict_proba(vec)[0]))
            p_spam = proba[SPAM]

            if pred == SPAM:
                st.error(f"🚨 Ye SPAM hai (confidence: {p_spam*100:.1f}%)")
            else:
                st.success(f"✅ Ye SPAM NAHI hai (confidence: {proba[NOT_SPAM]*100:.1f}%)")
            st.progress(float(p_spam), text=f"Spam probability: {p_spam*100:.1f}%")

# ---------------- TAB 2: DASHBOARD ----------------
with tab2:
    metrics, cm, top = compute_stats()
    label_names = df["label"].map({SPAM: "Spam", NOT_SPAM: "Not Spam"})

    st.subheader("Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Messages", f"{len(df):,}")
    m2.metric("Spam", f"{(df['label'] == SPAM).sum():,}")
    m3.metric("Not Spam", f"{(df['label'] == NOT_SPAM).sum():,}")
    m4.metric("Vocabulary Size", f"{len(cv.get_feature_names_out()):,}")

    st.subheader("Model Performance")
    p1, p2, p3, p4 = st.columns(4)
    for col, (k, v) in zip([p1, p2, p3, p4], metrics.items()):
        col.metric(k, f"{v*100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Class Distribution")
        st.bar_chart(label_names.value_counts())
    with right:
        st.subheader("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(cm,
                         index=["Actual Spam", "Actual Not Spam"],
                         columns=["Predicted Spam", "Predicted Not Spam"]),
            use_container_width=True,
        )

    st.subheader("Message Length (words)")
    lengths = df["text"].str.split().str.len()
    avg_len = lengths.groupby(label_names).mean().round(1)
    st.bar_chart(avg_len)

    st.subheader("Top 10 Words")
    w1, w2 = st.columns(2)
    with w1:
        st.write("**Spam messages**")
        st.bar_chart(top["Spam"])
    with w2:
        st.write("**Not spam messages**")
        st.bar_chart(top["Not Spam"])

st.caption("Built with Python, Scikit-learn, CountVectorizer & Streamlit")