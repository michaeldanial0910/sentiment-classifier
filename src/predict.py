"""Inference for Arabic/English customer feedback.

CLI:
    python -m src.predict "الغرفة نظيفة لكن الإفطار سيء"
    python -m src.predict --model transformer "great location, rude staff"
    python -m src.predict --file reviews.txt            # one review per line -> CSV on stdout

Models:
    tfidf        models/tfidf_lr.joblib (always available after `python -m src.train_classical`)
    transformer  models/xlmr_finetuned/ if present (fine-tuned in Colab), otherwise the public pretrained
                 checkpoint cardiffnlp/twitter-xlm-roberta-base-sentiment (downloads ~1.1 GB on first use).
                 Needs: pip install torch transformers
"""
import argparse
import functools
import json
import sys

import joblib
import numpy as np
import pandas as pd

from src.clean import clean_for_tfidf, detect_language, light_clean, strip_booking_score_title
from src.config import LABELS, MODELS, REPORTS

PRETRAINED = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


@functools.lru_cache(maxsize=1)
def _thresholds() -> dict:
    """Confidence below which a prediction goes to a person (picked on val by src/error_analysis.py)."""
    path = REPORTS / "triage_thresholds.json"
    return json.loads(path.read_text())["thresholds"] if path.exists() else {}


@functools.lru_cache(maxsize=1)
def _tfidf():
    return joblib.load(MODELS / "tfidf_lr.joblib")


@functools.lru_cache(maxsize=1)
def _transformer():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("Transformer mode needs `pip install torch transformers`.") from e
    local = MODELS / "xlmr_finetuned"
    source = str(local) if (local / "config.json").exists() else PRETRAINED
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source).eval()
    name = "C · XLM-R fine-tuned" if source == str(local) else "B · XLM-R pretrained (no training)"
    return tok, model, name


def available_transformer_name() -> str:
    local = MODELS / "xlmr_finetuned" / "config.json"
    return "C · XLM-R fine-tuned" if local.exists() else "B · XLM-R pretrained (no training)"


def _prep(texts):
    texts = [light_clean(strip_booking_score_title(t)) for t in texts]
    langs = [detect_language(t) for t in texts]
    return texts, langs


def predict_tfidf(texts) -> pd.DataFrame:
    texts, langs = _prep(texts)
    model = _tfidf()
    proba = model.predict_proba([clean_for_tfidf(t, l) for t, l in zip(texts, langs)])
    proba = proba[:, [list(model.classes_).index(l) for l in LABELS]]
    return _frame(texts, langs, proba, "A · TF-IDF + LogReg", _thresholds().get("tfidf_lr", 0.6))


def predict_transformer(texts, max_len=256) -> pd.DataFrame:
    import torch
    texts, langs = _prep(texts)
    tok, model, name = _transformer()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    if set(id2label.values()) != set(LABELS):
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
    order = [next(k for k, v in id2label.items() if v == l) for l in LABELS]
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        proba = torch.softmax(model(**enc).logits, -1)[:, order].numpy()
    key = "finetuned_xlmr" if name.startswith("C") else "zeroshot_xlmr"
    return _frame(texts, langs, proba, name, _thresholds().get(key, 0.6))


def _frame(texts, langs, proba, name, threshold) -> pd.DataFrame:
    out = pd.DataFrame(proba.round(4), columns=[f"p_{l}" for l in LABELS])
    out.insert(0, "confidence", proba.max(1).round(4))
    out.insert(0, "sentiment", np.array(LABELS)[proba.argmax(1)])
    out.insert(0, "lang", langs)
    out.insert(0, "text", texts)
    out["model"] = name
    # Low-confidence predictions are flagged for a human instead of being trusted blindly.
    out["needs_review"] = out["confidence"] < threshold
    return out


def predict(texts, model="tfidf") -> pd.DataFrame:
    return predict_tfidf(texts) if model == "tfidf" else predict_transformer(texts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="review text(s)")
    ap.add_argument("--file", help="text file, one review per line")
    ap.add_argument("--model", choices=["tfidf", "transformer"], default="tfidf")
    args = ap.parse_args(argv)
    texts = list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        ap.error("give at least one review, or --file")
    out = predict(texts, args.model)
    if args.file:
        out.to_csv(sys.stdout, index=False)
    else:
        for _, r in out.iterrows():
            flag = "  (low confidence: send to a human)" if r["needs_review"] else ""
            print(f"[{r['lang']}] {r['sentiment']:<8} {r['confidence']:.0%}{flag}  |  {r['text'][:80]}")


if __name__ == "__main__":
    main()
