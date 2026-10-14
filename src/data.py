"""Build the bilingual dataset and the one fixed split every approach is scored on.

    python -m src.data

Output: data/processed/reviews.parquet with columns
    id, lang, rating, label, text (light clean, for transformers), text_tfidf, split
and data/processed/split_summary.csv.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean import clean_for_tfidf, light_clean, strip_booking_score_title
from src.config import (ARABIC_FILE, ARABIC_SAMPLE_SIZE, ENGLISH_FILE, MIN_WORDS, PROCESSED,
                        RATING_TO_LABEL, SEED, SPLIT)


def load_arabic(path=ARABIC_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-16", quoting=3)
    df = df.rename(columns={"review": "raw_text"})[["rating", "raw_text"]]
    df["raw_text"] = df["raw_text"].astype(str)
    stripped = df["raw_text"].map(strip_booking_score_title)
    df["had_score_title"] = stripped != df["raw_text"]
    df["text"] = stripped.map(light_clean)
    df["lang"] = "ar"
    return df


def load_english(path=ENGLISH_FILE) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"Review": "raw_text", "Rating": "rating"})
    df["raw_text"] = df["raw_text"].astype(str)
    df["had_score_title"] = False
    df["text"] = df["raw_text"].map(light_clean)
    df["lang"] = "en"
    return df


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Map ratings to labels, drop very short and duplicate reviews (within a language)."""
    df = df[df["rating"].isin(RATING_TO_LABEL)].copy()
    df["rating"] = df["rating"].astype(int)
    df["label"] = df["rating"].map(RATING_TO_LABEL)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    # Same text with different ratings would be contradictory labels; keep the first.
    df = df.drop_duplicates(subset=["lang", "text"])
    return df


def sample_arabic(ar: pd.DataFrame, n=ARABIC_SAMPLE_SIZE) -> pd.DataFrame:
    """Stratified random sample that keeps the natural Arabic label mix."""
    keep, _ = train_test_split(ar.index, train_size=n, stratify=ar["label"], random_state=SEED)
    return ar.loc[keep].reset_index(drop=True)


def split(df: pd.DataFrame) -> pd.DataFrame:
    """70/15/15, stratified by language x label so every split has the same mix."""
    strat = df["lang"] + "_" + df["label"]
    train, rest = train_test_split(df.index, test_size=1 - SPLIT[0], stratify=strat, random_state=SEED)
    val, test = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                                 stratify=strat.loc[rest], random_state=SEED)
    df["split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    return df


def build() -> pd.DataFrame:
    ar = tidy(load_arabic())
    en = tidy(load_english())
    ar = sample_arabic(ar)
    df = pd.concat([ar, en], ignore_index=True)
    df["text_tfidf"] = [clean_for_tfidf(t, l) for t, l in zip(df["text"], df["lang"])]
    df = split(df)
    df.insert(0, "id", [f"{l}_{i:05d}" for i, l in enumerate(df["lang"])])
    cols = ["id", "lang", "rating", "label", "text", "text_tfidf", "split", "had_score_title"]
    return df[cols]


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(PROCESSED / "reviews.parquet", index=False)
    # Plain CSV of the test/val/train text too: this is what gets uploaded to Colab.
    df[["id", "lang", "label", "text", "split"]].to_csv(PROCESSED / "reviews_for_colab.csv", index=False)
    summary = pd.crosstab([df["split"], df["lang"]], df["label"], margins=True)
    summary.to_csv(PROCESSED / "split_summary.csv")
    print(summary)
    print(f"\nSaved {len(df):,} reviews to {PROCESSED / 'reviews.parquet'}")
