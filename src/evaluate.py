"""Score every approach with the same code on the same test rows.

    python -m src.evaluate

Reads reports/predictions/<model>_test.csv for every model in MODELS that has one.
Writes:
    reports/metrics.csv              per model x scope (all/ar/en): macro-F1 with 95% CI, accuracy, per-class F1
    reports/pairwise_differences.csv paired bootstrap of the macro-F1 difference between models
    reports/cost.csv                 latency and model size
    reports/figures/*.png            score chart and confusion matrices
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import FIGURES, LABELS, MODELS as MODEL_DIR, PREDICTIONS, PROCESSED, REPORTS, SEED

MODELS = {   # file prefix -> display name
    "tfidf_lr": "A · TF-IDF + LogReg",
    "zeroshot_xlmr": "B · XLM-R pretrained (no training)",
    "finetuned_xlmr": "C · XLM-R fine-tuned",
}
SCOPES = {"all": None, "ar": "ar", "en": "en"}
N_BOOT = 2000


def load(split="test") -> tuple[pd.DataFrame, dict]:
    """Gold labels plus each available model's predictions, aligned on id."""
    gold = pd.read_parquet(PROCESSED / "reviews.parquet")
    gold = gold[gold["split"] == split][["id", "lang", "label", "text"]].reset_index(drop=True)
    preds = {}
    for key in MODELS:
        path = PREDICTIONS / f"{key}_{split}.csv"
        if not path.exists():
            continue
        p = pd.read_csv(path).set_index("id")
        missing = set(gold["id"]) - set(p.index)
        assert not missing, f"{key}: {len(missing)} test ids missing; predictions are not on the same rows"
        preds[key] = p.loc[gold["id"]].reset_index()
    return gold, preds


_CODE = {l: i for i, l in enumerate(LABELS)}


def macro_f1(y, p) -> float:
    """Macro-F1 over the 3 labels (same result as sklearn's f1_score(average="macro", zero_division=0),
    written with numpy because the bootstrap calls it ~50,000 times)."""
    y = np.fromiter((_CODE[v] for v in y), int, len(y))
    p = np.fromiter((_CODE[v] for v in p), int, len(p))
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def bootstrap_idx(n, rng):
    return [rng.integers(0, n, n) for _ in range(N_BOOT)]


def scope_mask(gold, scope):
    return np.ones(len(gold), bool) if SCOPES[scope] is None else (gold["lang"] == SCOPES[scope]).to_numpy()


def metrics_table(gold, preds) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for scope in SCOPES:
        m = scope_mask(gold, scope)
        y = gold["label"].to_numpy()[m]
        idx = bootstrap_idx(m.sum(), rng)
        for key, p in preds.items():
            yp = p["pred"].to_numpy()[m]
            boots = [macro_f1(y[i], yp[i]) for i in idx]
            per_class = f1_score(y, yp, labels=LABELS, average=None, zero_division=0)
            rows.append({"model": key, "name": MODELS[key], "scope": scope, "n": int(m.sum()),
                         "macro_f1": macro_f1(y, yp),
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "accuracy": accuracy_score(y, yp),
                         **{f"f1_{l}": v for l, v in zip(LABELS, per_class)}})
    return pd.DataFrame(rows)


def pairwise(gold, preds) -> pd.DataFrame:
    """Paired bootstrap: resample the same rows for both models, look at the difference."""
    rng = np.random.default_rng(SEED + 1)
    keys = list(preds)
    rows = []
    for scope in SCOPES:
        m = scope_mask(gold, scope)
        y = gold["label"].to_numpy()[m]
        idx = bootstrap_idx(m.sum(), rng)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pa, pb = preds[a]["pred"].to_numpy()[m], preds[b]["pred"].to_numpy()[m]
                diffs = np.array([macro_f1(y[j], pb[j]) - macro_f1(y[j], pa[j]) for j in idx])
                rows.append({"scope": scope, "model_a": a, "model_b": b,
                             "diff_b_minus_a": macro_f1(y, pb) - macro_f1(y, pa),
                             "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
                             "share_b_better": float((diffs > 0).mean())})
    return pd.DataFrame(rows)


def cost_table(preds) -> pd.DataFrame:
    rows = []
    for key in preds:
        lat_path = PREDICTIONS / f"{key}_latency.json"
        lat = json.loads(lat_path.read_text()) if lat_path.exists() else {}
        if key == "tfidf_lr":
            size_mb = (MODEL_DIR / "tfidf_lr.joblib").stat().st_size / 1e6 if (MODEL_DIR / "tfidf_lr.joblib").exists() else np.nan
            train = "≈1 min, laptop CPU"
        else:
            size_mb = 1112.0   # XLM-R base, 278M params, fp32 safetensors
            log = PREDICTIONS / "finetuned_xlmr_training_log.json"
            train = "none" if key == "zeroshot_xlmr" else (
                f"{json.loads(log.read_text())['train_minutes']} min, T4 GPU" if log.exists() else "T4 GPU")
        ms = lat.get("ms_per_review_batch_cpu", lat.get("ms_per_review_batch_gpu"))
        rows.append({"model": key, "name": MODELS[key], "ms_per_review": ms, "hardware": lat.get("hardware", ""),
                     "size_mb": round(size_mb, 1), "training": train})
    return pd.DataFrame(rows)


def plot_scores(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    keys = list(dict.fromkeys(table["model"]))
    colors = {"tfidf_lr": "#8d99ae", "zeroshot_xlmr": "#f4a261", "finetuned_xlmr": "#2a9d8f"}
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        t = table[table["model"] == key].set_index("scope").loc[list(SCOPES)]
        x = np.arange(len(SCOPES)) + (i - (len(keys) - 1) / 2) * width
        ax.bar(x, t["macro_f1"], width, color=colors[key], label=MODELS[key],
               yerr=[t["macro_f1"] - t["ci_low"], t["ci_high"] - t["macro_f1"]], capsize=3)
        for xi, v in zip(x, t["macro_f1"]):
            ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(SCOPES)), ["Both languages", "Arabic", "English"])
    ax.set_ylabel("macro-F1 (test)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left", ncol=len(keys), frameon=False, bbox_to_anchor=(0, 1.13))
    ax.set_title("Test macro-F1 with 95% bootstrap CI", pad=26)
    plt.tight_layout(); plt.savefig(FIGURES / "model_comparison.png", dpi=150); plt.close()


def plot_confusions(gold, preds):
    fig, axes = plt.subplots(len(preds), 2, figsize=(7.2, 3.1 * len(preds)), squeeze=False)
    short = ["neg", "neu", "pos"]
    for r, (key, p) in enumerate(preds.items()):
        for c, lang in enumerate(("ar", "en")):
            m = (gold["lang"] == lang).to_numpy()
            cm = confusion_matrix(gold["label"][m], p["pred"][m], labels=LABELS)
            share = cm / cm.sum(1, keepdims=True)
            ax = axes[r, c]
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if share[i, j] > 0.6 else "black")
            ax.set_xticks(range(3), short); ax.set_yticks(range(3), short)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{MODELS[key].split(' · ')[0]} · {'Arabic' if lang == 'ar' else 'English'}", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGURES / "confusion_matrices.png", dpi=150); plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    gold, preds = load("test")
    print("Models with test predictions:", list(preds))
    table = metrics_table(gold, preds)
    table.to_csv(REPORTS / "metrics.csv", index=False)
    show = table[["name", "scope", "macro_f1", "ci_low", "ci_high", "accuracy"] + [f"f1_{l}" for l in LABELS]]
    print(show.round(3).to_string(index=False))
    if len(preds) > 1:
        pw = pairwise(gold, preds)
        pw.to_csv(REPORTS / "pairwise_differences.csv", index=False)
        print(pw.round(3).to_string(index=False))
    cost = cost_table(preds)
    cost.to_csv(REPORTS / "cost.csv", index=False)
    print(cost.to_string(index=False))
    plot_scores(table)
    plot_confusions(gold, preds)


if __name__ == "__main__":
    main()
