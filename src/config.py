"""Paths, label mapping and run settings shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PREDICTIONS = REPORTS / "predictions"

ARABIC_FILE = RAW / "hard_unbalanced_reviews.txt"      # HARD, UTF-16 TSV
ENGLISH_FILE = RAW / "tripadvisor_hotel_reviews.csv"   # TripAdvisor 20k

SEED = 42
LABELS = ["negative", "neutral", "positive"]           # fixed order everywhere
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Star rating -> sentiment. Same mapping for both languages.
RATING_TO_LABEL = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

# Arabic is sampled down to roughly the English size so neither language dominates
# training or the overall metric. The natural label mix of each language is kept.
ARABIC_SAMPLE_SIZE = 20_000
MIN_WORDS = 3            # drop reviews shorter than this after cleaning
SPLIT = (0.70, 0.15, 0.15)

# Booking.com fills the review title with a score word when the guest writes none
# ("استثنائي" = exceptional -> always 5 stars). That word *is* the label, so it is
# stripped before anything else (see notebooks/01_eda.ipynb, "label leakage").
BOOKING_SCORE_WORDS = [
    "ضعيف جداً", "ضعيف جدا", "ضعيف", "مخيب للأمل", "مقبول", "مرضي", "حسن",
    "جيد جداً", "جيد جدا", "جيد", "رائع", "ممتاز", "استثنائي",
]
