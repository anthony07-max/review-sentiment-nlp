"""Sentiment models (lexicon baseline vs TF-IDF + logistic regression), NMF topics, extractive summaries, charts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from summarise import summarise

ROOT = Path(__file__).resolve().parents[1]
P, C, M = ROOT / "data" / "processed", ROOT / "charts", ROOT / "models"
for d in (C, M): d.mkdir(exist_ok=True)
LABELS = ["negative", "neutral", "positive"]

def main():
    d = pd.read_parquet(P / "reviews.parquet")
    tr, te = train_test_split(d, test_size=0.2, stratify=d.stars, random_state=42)

    # 1) Lexicon baseline (VADER) - no training
    sia = SentimentIntensityAnalyzer()
    comp = te.text.map(lambda t: sia.polarity_scores(t)["compound"])
    vader = pd.cut(comp, [-1.01, -0.05, 0.6, 1.01], labels=LABELS).astype(str)

    # 2) TF-IDF + logistic regression (3-class) and star regression via 5-class
    pipe3 = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)),
                      ("clf", LogisticRegression(max_iter=2000, C=4))]).fit(tr.text, tr.sentiment.astype(str))
    pred3 = pipe3.predict(te.text)
    pipe5 = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)),
                      ("clf", LogisticRegression(max_iter=2000, C=4))]).fit(tr.text, tr.stars)
    pred5 = pipe5.predict(te.text)
    proba5 = pipe5.predict_proba(te.text)
    exp_stars = proba5 @ pipe5.classes_
    joblib.dump(pipe3, M / "sentiment_3class.joblib")

    y = te.sentiment.astype(str)
    res = pd.DataFrame({
        "model": ["VADER lexicon (no training)", "TF-IDF + logistic regression"],
        "accuracy_3class": [accuracy_score(y, vader), accuracy_score(y, pred3)],
        "macro_f1_3class": [f1_score(y, vader, average="macro"), f1_score(y, pred3, average="macro")]})
    res.to_csv(P / "sentiment_results.csv", index=False)
    star = {"exact_star_accuracy": accuracy_score(te.stars, pred5), "within_one_star": float((abs(pred5 - te.stars) <= 1).mean()),
            "mae_expected_stars": mean_absolute_error(te.stars, exp_stars)}
    print(res); print(star)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, (name, p) in zip(ax, [("VADER", vader), ("TF-IDF + LR", pred3)]):
        cm = confusion_matrix(y, p, labels=LABELS, normalize="true")
        a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3): a.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center")
        a.set_xticks(range(3), LABELS); a.set_yticks(range(3), LABELS); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
        a.set_title(f"{name}: macro F1 {f1_score(y, p, average='macro'):.2f}")
    fig.tight_layout(); fig.savefig(C / "01_confusion_matrices.png", dpi=150); plt.close()

    # Strongest words/phrases
    vec, clf = pipe3.named_steps["tfidf"], pipe3.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    ci = {c: i for i, c in enumerate(clf.classes_)}
    w = pd.DataFrame({"term": vocab, "coef": clf.coef_[ci["positive"]] - clf.coef_[ci["negative"]]})
    top_pos, top_neg = w.nlargest(15, "coef"), w.nsmallest(15, "coef")
    fig, ax = plt.subplots(figsize=(8, 7))
    t = pd.concat([top_neg, top_pos[::-1]])
    ax.barh(t.term, t.coef, color=np.where(t.coef > 0, "#2a9d8f", "#e76f51")); ax.set_title("Terms that push a review positive vs negative")
    fig.tight_layout(); fig.savefig(C / "02_sentiment_terms.png", dpi=150); plt.close()

    # 3) Topics in negative and positive reviews (NMF)
    topics = {}
    for label, k in [("negative", 10), ("positive", 10)]:
        sub = d[d.sentiment == label]
        tv = TfidfVectorizer(stop_words="english", max_df=0.4, min_df=10, ngram_range=(1, 2), max_features=20_000)
        X = tv.fit_transform(sub.text)
        nmf = NMF(n_components=k, random_state=0, init="nndsvd", max_iter=400)
        W = nmf.fit_transform(X)
        terms = np.array(tv.get_feature_names_out())
        dom = W.argmax(1)
        rows = []
        for i in range(k):
            members = sub.text.values[(dom == i) & (W.max(1) > 0.02)]
            top_docs = sub.text.values[np.argsort(-W[:, i])[:400]]
            rows.append({"topic": i, "top_terms": ", ".join(terms[np.argsort(-nmf.components_[i])[:10]]),
                         "share": float((dom == i).mean()), "n_reviews": int(len(members)),
                         "mean_stars": float(sub.stars.values[dom == i].mean()),
                         "summary": summarise(top_docs, k=3)})
        topics[label] = sorted(rows, key=lambda r: -r["share"])
        tdf = pd.DataFrame(topics[label]); tdf["summary"] = tdf.summary.map(" | ".join)
        tdf.to_csv(P / f"topics_{label}.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        tt = tdf.sort_values("share")
        ax.barh([", ".join(s.split(", ")[:4]) for s in tt.top_terms], tt.share * 100, color="#e76f51" if label == "negative" else "#2a9d8f")
        ax.set_xlabel("% of reviews (dominant topic)"); ax.set_title(f"Topics in {label} reviews (NMF, k={k})")
        fig.tight_layout(); fig.savefig(C / f"0{3 if label == 'negative' else 4}_topics_{label}.png", dpi=150); plt.close()

    # Length vs stars
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d.groupby("stars").words.median().plot.bar(ax=ax, color="#457b9d", rot=0)
    ax.set_ylabel("Median words"); ax.set_title("Unhappy customers write more"); fig.tight_layout(); fig.savefig(C / "05_length_by_stars.png", dpi=150); plt.close()

    kf = {"n_reviews": int(len(d)), "test_size": int(len(te)), "sentiment": res.round(3).to_dict("records"), "stars": {k: round(v, 3) for k, v in star.items()},
          "median_words_by_star": d.groupby("stars").words.median().to_dict(),
          "top_positive_terms": top_pos.term.tolist(), "top_negative_terms": top_neg.term.tolist(),
          "topics": topics}
    (P / "key_findings.json").write_text(json.dumps(kf, indent=2, default=float))
    for lab in topics:
        print(lab); [print(f"  {r['share']:.1%} {r['top_terms']}\n     -> {r['summary'][0][:150]}") for r in topics[lab]]

if __name__ == "__main__":
    main()
