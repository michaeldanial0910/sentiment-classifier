"""Error analysis on the test split.

    python -m src.error_analysis

Writes:
    reports/error_examples.csv      every misclassified test review, per model, most confident mistakes first
    reports/errors_by_length.csv    macro-F1 by language x review-length bucket
    reports/agreement.csv           how often the models are right/wrong together
    reports/tfidf_top_features.csv strongest n-grams per class in the TF-IDF model (what the baseline relies on)
    reports/confidence_triage.csv   auto-handle confident predictions, route the rest to a person
    reports/triage_thresholds.json  per-model confidence threshold picked on val (used by src/predict.py)
    reports/figures/f1_by_length.png
"""
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES, LABELS, MODELS as MODEL_DIR, REPORTS
from src.evaluate import MODELS, load, macro_f1

LENGTH_BINS = {"ar": [0, 8, 15, 30, 10_000], "en": [0, 50, 90, 150, 10_000]}


def error_examples(gold, preds) -> pd.DataFrame:
    frames = []
    for key, p in preds.items():
        conf = p[[f"p_{l}" for l in LABELS]].max(axis=1)
        wrong = p["pred"] != gold["label"]
        frames.append(pd.DataFrame({"model": key, "id": gold["id"], "lang": gold["lang"], "true": gold["label"],
                                    "pred": p["pred"], "confidence": conf.round(3), "text": gold["text"]})[wrong])
    return pd.concat(frames).sort_values(["model", "confidence"], ascending=[True, False])


def by_length(gold, preds) -> pd.DataFrame:
    rows = []
    n_words = gold["text"].str.split().str.len()
    for lang, bins in LENGTH_BINS.items():
        m = (gold["lang"] == lang).to_numpy()
        bucket = pd.cut(n_words[m], bins=bins, right=False,
                        labels=[f"{a}-{b - 1}" if b < 10_000 else f"{a}+" for a, b in zip(bins, bins[1:])])
        for b in bucket.cat.categories:
            mb = (bucket == b).to_numpy()
            y = gold["label"].to_numpy()[m][mb]
            for key, p in preds.items():
                rows.append({"lang": lang, "words": b, "n": int(mb.sum()), "model": key,
                             "macro_f1": macro_f1(y, p["pred"].to_numpy()[m][mb])})
    return pd.DataFrame(rows)


def agreement(gold, preds) -> pd.DataFrame:
    right = pd.DataFrame({k: (p["pred"] == gold["label"]).to_numpy() for k, p in preds.items()})
    right["lang"] = gold["lang"].to_numpy()
    pattern = right.drop(columns="lang").apply(lambda r: "".join("✓" if v else "✗" for v in r), axis=1)
    out = pd.crosstab(pattern, right["lang"], margins=True)
    out.index.name = "correct? (" + ", ".join(preds) + ")"
    return out


def tfidf_top_features(k=15) -> pd.DataFrame:
    path = MODEL_DIR / "tfidf_lr.joblib"
    if not path.exists():
        return pd.DataFrame()
    model = joblib.load(path)
    names = model.named_steps["features"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_
    rows = []
    for ci, cls in enumerate(model.named_steps["clf"].classes_):
        word = np.array([n.startswith("word__") for n in names])
        order = np.argsort(-coef[ci] * word)[:k]           # word n-grams are readable; char n-grams are not
        rows += [{"class": cls, "rank": r + 1, "ngram": names[j].removeprefix("word__"), "weight": round(coef[ci, j], 3)}
                 for r, j in enumerate(order)]
    return pd.DataFrame(rows)


TARGET_ACCURACY = 0.90   # accuracy we want on the reviews the model handles alone


def triage(gold_val, preds_val, gold_test, preds_test) -> tuple[pd.DataFrame, dict]:
    """Pick, on val, the lowest confidence threshold whose auto-handled reviews reach TARGET_ACCURACY.
    Then report what that threshold does on test (share auto-handled, accuracy there)."""
    rows, chosen = [], {}
    for key in preds_test:
        if key not in preds_val:
            continue
        conf_v = preds_val[key][[f"p_{l}" for l in LABELS]].max(axis=1).to_numpy()
        ok_v = (preds_val[key]["pred"] == gold_val["label"]).to_numpy()
        grid = np.round(np.arange(0.40, 0.99, 0.01), 2)
        acc = [ok_v[conf_v >= t].mean() if (conf_v >= t).any() else np.nan for t in grid]
        hit = [t for t, a in zip(grid, acc) if a >= TARGET_ACCURACY]
        t = float(hit[0]) if hit else float(grid[-1])
        chosen[key] = t
        conf_t = preds_test[key][[f"p_{l}" for l in LABELS]].max(axis=1).to_numpy()
        ok_t = (preds_test[key]["pred"] == gold_test["label"]).to_numpy()
        for scope, m in (("all", np.ones(len(gold_test), bool)), ("ar", (gold_test["lang"] == "ar").to_numpy()),
                         ("en", (gold_test["lang"] == "en").to_numpy())):
            auto = m & (conf_t >= t)
            rows.append({"model": key, "threshold": t, "scope": scope,
                         "auto_handled_share": auto.sum() / m.sum(),
                         "accuracy_auto_handled": ok_t[auto].mean() if auto.any() else np.nan,
                         "accuracy_all": ok_t[m].mean(),
                         "errors_caught_by_review": 1 - (~ok_t[auto]).sum() / max((~ok_t[m]).sum(), 1)})
    return pd.DataFrame(rows), chosen


def plot_length(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    colors = {"tfidf_lr": "#8d99ae", "zeroshot_xlmr": "#f4a261", "finetuned_xlmr": "#2a9d8f"}
    for ax, lang in zip(axes, ("ar", "en")):
        d = df[df["lang"] == lang]
        for key, g in d.groupby("model", sort=False):
            ax.plot(g["words"].astype(str), g["macro_f1"], marker="o", color=colors[key], label=MODELS[key])
        ax.set_title("Arabic" if lang == "ar" else "English"); ax.set_xlabel("words in review")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1 (test)")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout(); plt.savefig(FIGURES / "f1_by_length.png", dpi=150); plt.close()


def main():
    gold, preds = load("test")
    err = error_examples(gold, preds)
    err.to_csv(REPORTS / "error_examples.csv", index=False)
    print(err.groupby(["model", "lang", "true", "pred"]).size().unstack("pred", fill_value=0))

    length = by_length(gold, preds)
    length.to_csv(REPORTS / "errors_by_length.csv", index=False)
    print(length.pivot_table(index=["lang", "words"], columns="model", values="macro_f1", observed=True).round(3))
    plot_length(length)

    if len(preds) > 1:
        ag = agreement(gold, preds)
        ag.to_csv(REPORTS / "agreement.csv")
        print(ag)

    gold_val, preds_val = load("val")
    tri, chosen = triage(gold_val, preds_val, gold, preds)
    tri.to_csv(REPORTS / "confidence_triage.csv", index=False)
    (REPORTS / "triage_thresholds.json").write_text(json.dumps(
        {"target_accuracy_on_auto_handled": TARGET_ACCURACY, "chosen_on": "val", "thresholds": chosen}, indent=2))
    print(tri.round(3).to_string(index=False))

    top = tfidf_top_features()
    top.to_csv(REPORTS / "tfidf_top_features.csv", index=False)
    print(top.groupby("class")["ngram"].apply(lambda s: " | ".join(s[:10])).to_string())


if __name__ == "__main__":
    main()
