"""Text cleaning.

Two levels, on purpose:
- light_clean    : whitespace/URL tidy only. Fed to the transformers, whose tokenizers were
                   trained on raw text (normalising Arabic letters would hurt them).
- clean_for_tfidf: language-specific normalisation that shrinks the vocabulary for the
                   bag-of-words baseline (Arabic letter variants, diacritics, elongation).
"""
import re

from src.config import BOOKING_SCORE_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")  # tashkeel + Quranic marks
_TATWEEL = re.compile("ـ")
_REPEAT = re.compile(r"(.)\1{2,}")  # "رااااائع" / "soooo" -> 2 copies

# Leading Booking.com score word: optional stray quote, the word, then ". "
_SCORE_PREFIX = re.compile(
    r'^\s*"?(' + "|".join(sorted(map(re.escape, BOOKING_SCORE_WORDS), key=len, reverse=True)) + r")\.\s*"
)


def detect_language(text: str) -> str:
    """'ar' if at least 30% of the letters are Arabic, else 'en'. Good enough for two languages."""
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum(bool(_ARABIC_CHAR.match(c)) for c in letters)
    return "ar" if arabic / len(letters) >= 0.3 else "en"


def strip_booking_score_title(text: str) -> str:
    """Remove the auto-generated score word Booking.com puts in front of untitled reviews.

    User-written titles are wrapped in “…” and are kept; only the unquoted score word goes.
    """
    return _SCORE_PREFIX.sub("", str(text), count=1)


def light_clean(text: str) -> str:
    text = _URL.sub(" ", str(text))
    text = text.replace("“", '"').replace("”", '"')
    return _WS.sub(" ", text).strip(" .")


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text


def clean_for_tfidf(text: str, lang: str) -> str:
    text = light_clean(text)
    if lang == "ar":
        text = normalize_arabic(text)
    text = text.lower()
    text = _REPEAT.sub(r"\1\1", text)
    text = re.sub(r"[\"'“”«»()\[\]]", " ", text)
    return _WS.sub(" ", text).strip()
