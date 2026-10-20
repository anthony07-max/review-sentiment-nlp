"""Tidy the reviews: stars 1-5, sentiment class, unescaped text, length."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    d = pd.read_parquet(ROOT / "data" / "raw" / "yelp_reviews.parquet")
    d["stars"] = d["label"] + 1  # HF labels are 0-4
    d["text"] = d.text.str.replace(r"\\n", " ", regex=True).str.replace(r'\\"', '"', regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    d["sentiment"] = pd.cut(d.stars, [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
    d["words"] = d.text.str.split().str.len()
    d = d[d.words >= 3].drop(columns="label").reset_index(drop=True)
    d.to_parquet(ROOT / "data" / "processed" / "reviews.parquet", index=False)
    print(len(d), d.stars.value_counts().sort_index().to_dict())

if __name__ == "__main__":
    main()
