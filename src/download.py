"""Download the Yelp Review Full test split (50,000 real Yelp reviews with 1-5 star ratings) from Hugging Face."""
from pathlib import Path
import requests

URL = "https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/refs%2Fconvert%2Fparquet/yelp_review_full/test/0000.parquet"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=300); r.raise_for_status()
    (RAW / "yelp_reviews.parquet").write_bytes(r.content)
    print("saved", len(r.content) // 1_000_000, "MB")

if __name__ == "__main__":
    main()
