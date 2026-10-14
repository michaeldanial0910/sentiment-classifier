"""Approach A: TF-IDF (word + character n-grams) -> Logistic Regression.

    python -m src.train_classical

One model for both languages, like the transformers. Hyperparameters are picked on the
validation split by macro-F1; the test split is only scored once, at the end.
Outputs:
    models/tfidf_lr.joblib
    reports/predictions/tfidf_lr_{val,test}.csv   (id, pred, p_negative, p_neutral, p_positive)
    reports/tfidf_search.csv                      (every config tried, with val scores)
"""
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import LABELS, MODELS, PREDICTIONS, PROCESSED, REPORTS, SEED

SEARCH = {
    "C": [0.125, 0.25, 0.5, 2.0, 8.0],
    "class_weight": [None, "balanced"],
    "char_ngrams": [(2, 5)],
    "min_df": [2],
}


def make_pipeline(C=2.0, class_weight="balanced", char_ngrams=(2, 5), min_df=2) -> Pipeline:
    features = FeatureUnion([
        # Word uni+bigrams catch phrases like "not clean" / "غير نظيف".
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df,
                                 sublinear_tf=True, max_features=200_000)),
        # Character n-grams within word boundaries: robust to Arabic prefixes/suffixes
        # (ال، و، ـها) and to typos/dialect spellings in both languages.
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df,
                                 sublinear_tf=True, max_features=300_000)),
    ])
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def predict_frame(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df["text_tfidf"])
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.classes_])
    out = out[[f"p_{l}" for l in LABELS]]
    out.insert(0, "pred", np.array(LABELS)[out.to_numpy().argmax(1)])
    out.insert(0, "id", df["id"].to_numpy())
    return out


def main():
    df = pd.read_parquet(PROCESSED / "reviews.parquet")
    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))

    rows = []
    for C, cw, cng, mdf in itertools.product(*SEARCH.values()):
        t0 = time.time()
        m = make_pipeline(C, cw, cng, mdf).fit(train["text_tfidf"], train["label"])
        pred = m.predict(val["text_tfidf"])
        row = {"C": C, "class_weight": str(cw), "char_ngrams": str(cng), "min_df": mdf,
               "val_macro_f1": f1_score(val["label"], pred, average="macro"),
               "fit_seconds": round(time.time() - t0, 1)}
        for lang in ("ar", "en"):
            mask = (val["lang"] == lang).to_numpy()
            row[f"val_macro_f1_{lang}"] = f1_score(val["label"][mask], pred[mask], average="macro")
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    search = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    search.to_csv(REPORTS / "tfidf_search.csv", index=False)
    best = search.iloc[0]
    params = {"C": float(best["C"]),
              "class_weight": None if best["class_weight"] == "None" else best["class_weight"],
              "char_ngrams": tuple(int(x) for x in best["char_ngrams"].strip("()").split(",")),
              "min_df": int(best["min_df"])}
    print("Best on val:", params)

    # Refit on train only (val stays a clean selection set, same as for the transformers).
    model = make_pipeline(**params).fit(train["text_tfidf"], train["label"])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "tfidf_lr.joblib", compress=3)
    (MODELS / "tfidf_lr_params.json").write_text(json.dumps(params, indent=2))

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", val), ("test", test)):
        t0 = time.time()
        predict_frame(model, part).to_csv(PREDICTIONS / f"tfidf_lr_{name}.csv", index=False)
        if name == "test":
            ms = (time.time() - t0) / len(part) * 1000
            (PREDICTIONS / "tfidf_lr_latency.json").write_text(json.dumps(
                {"ms_per_review_batch_cpu": round(ms, 3), "hardware": "CPU (batch of whole test set)"}))
    print("Saved model and predictions.")


if __name__ == "__main__":
    main()
