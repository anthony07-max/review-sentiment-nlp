import json
from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from summarise import summarise

st.set_page_config(page_title="Review sentiment & topics", layout="wide")
st.title("Customer review sentiment and topics")
st.caption("50,000 real Yelp reviews. TF-IDF + logistic regression, NMF topics, LLM-free extractive summaries. By Anthony Adusei-Poku (@anthony07-max)")
kf = json.loads((ROOT / "data" / "processed" / "key_findings.json").read_text())

t1, t2, t3 = st.tabs(["Try the model", "Topics", "Summarise your own reviews"])
with t1:
    mp = ROOT / "models" / "sentiment_3class.joblib"
    if mp.exists():
        model = joblib.load(mp)
        txt = st.text_area("Write a review", "The chicken was great but we waited 40 minutes and the server was rude.")
        p = pd.Series(model.predict_proba([txt])[0], index=model.classes_)
        st.bar_chart(p)
    else:
        st.warning("Run `bash run_pipeline.sh` first to train the model.")
    st.dataframe(pd.DataFrame(kf["sentiment"]), hide_index=True)
with t2:
    lab = st.radio("Reviews", ["negative", "positive"], horizontal=True)
    for r in kf["topics"][lab]:
        with st.expander(f"{r['share']:.1%} | {r['top_terms']}"):
            for s in r["summary"]:
                st.write("- " + s)
with t3:
    raw = st.text_area("Paste reviews, one per line", height=200)
    if raw.strip():
        for s in summarise([l for l in raw.splitlines() if l.strip()], k=3):
            st.write("- " + s)
