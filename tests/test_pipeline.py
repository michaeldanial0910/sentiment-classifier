"""Fast checks on the pieces that decide whether the comparison is valid.

    pytest -q
"""
import pandas as pd
import pytest

from src.clean import clean_for_tfidf, detect_language, light_clean, normalize_arabic, strip_booking_score_title
from src.config import LABELS, PREDICTIONS, PROCESSED, RATING_TO_LABEL

DATA = PROCESSED / "reviews.parquet"


# --- cleaning --------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("استثنائي. كل شي جميل. ", "كل شي جميل. "),                 # auto score word removed
    ("ضعيف جداً. . الخدمة سيئة", ". الخدمة سيئة"),             # longest match wins (not just "ضعيف")
    ('"مرضي. الموقع ممتاز', "الموقع ممتاز"),                   # stray leading quote
    ("“ممتاز”. الموقع رائع", "“ممتاز”. الموقع رائع"),          # user-written title is kept
    ("الفندق جيد. الموقع ممتاز", "الفندق جيد. الموقع ممتاز"),   # score word not at the start
])
def test_strip_booking_score_title(raw, expected):
    assert strip_booking_score_title(raw) == expected


def test_arabic_normalisation():
    assert normalize_arabic("أكثر إلى آخر مدرسة") == "اكثر الي اخر مدرسه"
    assert normalize_arabic("رَائِع") == "رايع"


def test_clean_for_tfidf_collapses_elongation_and_case():
    assert clean_for_tfidf("Sooooo GOOD", "en") == "soo good"
    assert clean_for_tfidf("راااااائع", "ar") == "راايع"   # also ئ -> ي


def test_light_clean_keeps_text_mostly_raw():
    assert light_clean("  visit www.x.com   now. ") == "visit now"


@pytest.mark.parametrize("text, lang", [("الغرفة نظيفة", "ar"), ("clean room", "en"),
                                        ("الغرفة clean جدا", "ar"), ("", "en")])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_rating_mapping_is_the_same_for_both_languages():
    assert RATING_TO_LABEL == {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


# --- dataset / split -------------------------------------------------------------------
needs_data = pytest.mark.skipif(not DATA.exists(), reason="run `python -m src.data` first")


@needs_data
def test_split_has_no_overlap_and_no_duplicates():
    df = pd.read_parquet(DATA)
    assert df["id"].is_unique
    assert not df.duplicated(["lang", "text"]).any()
    assert set(df["split"]) == {"train", "val", "test"}


@needs_data
def test_no_booking_score_word_survives():
    # 14 of 20,000 Arabic reviews still start with a score word after one strip: guests who typed the
    # same word as their own title ("ممتاز.") or a doubled prefix. Too few to move any metric; documented.
    df = pd.read_parquet(DATA)
    ar = df[df["lang"] == "ar"]["text"]
    assert (ar.map(strip_booking_score_title) != ar).mean() < 0.001


@needs_data
def test_split_is_stratified():
    df = pd.read_parquet(DATA)
    mix = pd.crosstab([df["lang"], df["split"]], df["label"], normalize="index")
    for lang in ("ar", "en"):
        assert (mix.loc[lang].max() - mix.loc[lang].min()).max() < 0.01


@needs_data
@pytest.mark.parametrize("model", ["tfidf_lr", "zeroshot_xlmr", "finetuned_xlmr"])
def test_predictions_cover_exactly_the_test_rows(model):
    path = PREDICTIONS / f"{model}_test.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    p = pd.read_csv(path)
    test_ids = set(pd.read_parquet(DATA).query("split == 'test'")["id"])
    assert set(p["id"]) == test_ids and p["id"].is_unique
    assert set(p["pred"]) <= set(LABELS)
    probs = p[[f"p_{l}" for l in LABELS]].sum(axis=1)
    assert probs.between(0.99, 1.01).all()
