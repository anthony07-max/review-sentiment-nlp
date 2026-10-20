"""LLM-free extractive summarisation: pick the most central sentences (TextRank-style on TF-IDF cosine similarity)."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def sentences(texts, min_words=6, max_words=40):
    out = []
    for t in texts:
        for s in re.split(r"(?<=[.!?])\s+", t):
            n = len(s.split())
            if min_words <= n <= max_words:
                out.append(s.strip())
    return out

def summarise(texts, k=3, max_sentences=3000, seed=0):
    sents = sentences(texts)
    if len(sents) > max_sentences:
        sents = list(np.random.default_rng(seed).choice(sents, max_sentences, replace=False))
    if len(sents) <= k:
        return sents
    X = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(sents)
    S = cosine_similarity(X); np.fill_diagonal(S, 0)
    # PageRank by power iteration
    row = S.sum(1, keepdims=True); M = np.divide(S, row, out=np.zeros_like(S), where=row > 0)
    r = np.ones(len(sents)) / len(sents)
    for _ in range(50):
        r = 0.15 / len(sents) + 0.85 * M.T @ r
    chosen = []
    for i in np.argsort(-r):
        if all(cosine_similarity(X[i], X[j])[0, 0] < 0.5 for j in chosen):  # avoid near-duplicates
            chosen.append(i)
        if len(chosen) == k:
            break
    return [sents[i] for i in chosen]
