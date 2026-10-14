import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (INK, INK_SOFT, LANG_NAME, MODEL_COLOR, MODEL_LETTER, MODEL_NAME, RULE, altair_theme, use_altair_theme,
                        figures, load_csv, load_test_predictions, need_data)
from src.config import LABELS

use_altair_theme()

st.title("Three ways to read a review, scored on the same 6,074")
st.html('<p class="lede">Every model was scored on the same held-out test reviews (3,000 Arabic, 3,074 English) '
        'by the same evaluation code. The metric is macro-F1, which counts the three classes equally, so a model '
        'can\'t look good by calling everything positive.</p>')

metrics = load_csv("metrics.csv")
pairs = load_csv("pairwise_differences.csv")
if metrics is None:
    st.warning("reports/metrics.csv is missing. Run `python -m src.evaluate`.")
    st.stop()

m = metrics.set_index(["model", "scope"])
diff = pairs.set_index(["scope", "model_a", "model_b"]).loc[("all", "tfidf_lr", "finetuned_xlmr")]
st.html(figures([
    (f"{m.loc[('finetuned_xlmr', 'all'), 'macro_f1']:.3f}", "macro-F1 of the fine-tuned transformer, the best model"),
    (f"+{diff['diff_b_minus_a']:.3f}", f"its lead over TF-IDF (95% CI +{diff['ci_low']:.3f} to +{diff['ci_high']:.3f})"),
    (f"{m.loc[('zeroshot_xlmr', 'all'), 'macro_f1']:.3f}", "the same transformer with no training on our data"),
    (f"{m.loc[('finetuned_xlmr', 'all'), 'f1_neutral']:.2f}", "best F1 on neutral, the hardest class"),
]))

# ---- forest plot: macro-F1 with 95% CI, one panel per scope ----------------------------
st.subheader("Macro-F1 with 95% bootstrap intervals")
d = metrics.copy()
d["Model"] = d["model"].map(lambda k: f"{MODEL_LETTER[k]}: {MODEL_NAME[k]}")
d["Scope"] = d["scope"].map(LANG_NAME)
order = [f"{MODEL_LETTER[k]}: {MODEL_NAME[k]}" for k in MODEL_NAME]
color = alt.Color("Model:N", scale=alt.Scale(domain=order, range=[MODEL_COLOR[k] for k in MODEL_NAME]),
                  legend=alt.Legend(orient="top", title=None))
base = alt.Chart(d).encode(y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
                           color=color)
tooltip = [alt.Tooltip("Model:N"), alt.Tooltip("Scope:N"), alt.Tooltip("macro_f1:Q", title="macro-F1", format=".3f"),
           alt.Tooltip("ci_low:Q", title="95% CI low", format=".3f"),
           alt.Tooltip("ci_high:Q", title="95% CI high", format=".3f"),
           alt.Tooltip("accuracy:Q", format=".1%")]
rule = base.mark_rule(strokeWidth=2).encode(x=alt.X("ci_low:Q", title="macro-F1", scale=alt.Scale(domain=[0.4, 0.8])),
                                            x2="ci_high:Q", tooltip=tooltip)
pts = base.mark_point(filled=True, size=110, opacity=1).encode(x="macro_f1:Q", tooltip=tooltip)
lbl = alt.Chart(d).mark_text(dy=-13, fontSize=12, color=INK).encode(
    y=alt.Y("Model:N", sort=order), x="macro_f1:Q", text=alt.Text("macro_f1:Q", format=".3f"))
forest = (rule + pts + lbl).properties(width=290, height=150).facet(
    column=alt.Column("Scope:N", sort=["Both languages", "Arabic", "English"], title=None))
st.altair_chart(forest, theme=None, width="content")
st.html('<p class="muted">Paired bootstrap, 2,000 resamples of the test reviews. The fine-tuned model beats the '
        'baseline in every resample; the gap is real but small.</p>')

# ---- per-class table -------------------------------------------------------------------
st.subheader("Per class")
scope = st.segmented_control("Language", ["all", "ar", "en"], default="all", format_func=LANG_NAME.get,
                             key="class_scope")
rows = []
t = metrics[metrics["scope"] == scope].set_index("model")
best = {c: t[c].max() for c in ["macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"]}
for k in MODEL_NAME:
    r = t.loc[k]
    cell = lambda c, f="{:.3f}": f'<td class="{"best" if r[c] == best[c] else ""}">{f.format(r[c])}</td>'
    rows.append(f'<tr><td><span class="dot" style="background:{MODEL_COLOR[k]}"></span> '
                f'{MODEL_LETTER[k]}: {MODEL_NAME[k]}</td>{cell("macro_f1")}{cell("accuracy", "{:.1%}")}'
                f'{cell("f1_negative")}{cell("f1_neutral")}{cell("f1_positive")}</tr>')
st.html('<table class="plain"><tr><th>Model</th><th>Macro-F1</th><th>Accuracy</th><th>F1 negative</th>'
        f'<th>F1 neutral</th><th>F1 positive</th></tr>{"".join(rows)}</table>')

# ---- confusion matrix ------------------------------------------------------------------
test = load_test_predictions()
st.subheader("Where each model goes wrong")
if test is None:
    need_data()
else:
    c1, c2 = st.columns([2, 1])
    model = c1.segmented_control("Model", list(MODEL_NAME), default="finetuned_xlmr",
                                 format_func=lambda k: f"{MODEL_LETTER[k]}: {MODEL_NAME[k]}", key="cm_model")
    lang = c2.segmented_control("Language ", ["ar", "en"], default="ar", format_func=LANG_NAME.get, key="cm_lang")
    sub = test[test["lang"] == lang]
    cm = pd.crosstab(sub["label"], sub[f"{model}_pred"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    long = cm.stack().rename("n").reset_index()
    long.columns = ["True", "Predicted", "n"]
    long["share"] = long["n"] / long.groupby("True")["n"].transform("sum")
    long["label"] = [f"{n:,}  ({sh:.0%})" for n, sh in zip(long["n"], long["share"])]
    heat = alt.Chart(long).encode(
        x=alt.X("Predicted:N", sort=LABELS, axis=alt.Axis(orient="top", labelAngle=0), title="predicted"),
        y=alt.Y("True:N", sort=LABELS, title="star rating says"),
        tooltip=["True", "Predicted", alt.Tooltip("n:Q", title="reviews"), alt.Tooltip("share:Q", format=".0%")])
    rect = heat.mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("share:Q", scale=alt.Scale(range=["#F1F3F8", "#24305E"], domain=[0, 1]), legend=None))
    txt = heat.mark_text(fontSize=13).encode(
        text=alt.Text("label:N"),
        color=alt.condition("datum.share > 0.5", alt.value("white"), alt.value(INK)))
    st.altair_chart((rect + txt).properties(width=520, height=260), theme=None, width="content")
    worst = long[long["True"] != long["Predicted"]].sort_values("n", ascending=False).iloc[0]
    st.html(f'<p class="muted">Biggest mistake here: {worst["n"]} {worst["True"]} reviews read as '
            f'{worst["Predicted"]} ({worst["share"]:.0%} of all {worst["True"]} ones).</p>')

# ---- triage simulator ------------------------------------------------------------------
st.subheader("How much can the model handle on its own?")
st.html('<p class="lede">A support team needs the easy reviews labelled and the hard ones flagged. Move the '
        'threshold: reviews the model is at least this sure about are labelled automatically, the rest go to a '
        'person.</p>')
if test is not None:
    thresholds = load_csv("confidence_triage.csv")
    c1, c2 = st.columns([1, 1.4])
    tmodel = c1.radio("Model", ["finetuned_xlmr", "tfidf_lr", "zeroshot_xlmr"],
                      format_func=lambda k: f"{MODEL_LETTER[k]}: {MODEL_NAME[k]}", key="tri_model")
    chosen = (thresholds[thresholds["model"] == tmodel]["threshold"].iloc[0]
              if thresholds is not None and tmodel in set(thresholds["model"]) else 0.8)
    t = c1.slider("Confidence threshold", 0.34, 0.99, float(min(chosen, 0.99)), 0.01, key=f"tri_t_{tmodel}")
    conf, ok = test[f"{tmodel}_conf"].to_numpy(), (test[f"{tmodel}_pred"] == test["label"]).to_numpy()
    auto = conf >= t
    c1.html(f'<p class="muted">Threshold picked on validation for 90% accuracy: {chosen:.2f}'
            f'{" (never reached for this model)" if tmodel == "zeroshot_xlmr" else ""}.</p>')
    c2.html(figures([
        (f"{auto.mean():.0%}", "of test reviews labelled automatically"),
        (f"{ok[auto].mean():.1%}" if auto.any() else "n/a", "accuracy on those"),
        (f"{(~auto).sum():,}", "reviews sent to a person"),
    ]))
    grid = np.round(np.arange(0.34, 0.995, 0.01), 2)
    curve = pd.DataFrame({"threshold": grid,
                          "auto-handled": [(conf >= g).mean() for g in grid],
                          "accuracy on auto-handled": [ok[conf >= g].mean() if (conf >= g).any() else np.nan
                                                       for g in grid]}).melt("threshold", var_name="measure")
    line = alt.Chart(curve).mark_line(strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="confidence threshold"),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        strokeDash=alt.StrokeDash("measure:N", legend=alt.Legend(orient="bottom", title=None)),
        color=alt.value(MODEL_COLOR[tmodel]),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "measure:N", alt.Tooltip("value:Q", format=".1%")])
    now = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(color=INK_SOFT, strokeDash=[2, 3]).encode(x="t:Q")
    c2.altair_chart((line + now).properties(height=230), theme=None, width="stretch")

# ---- cost --------------------------------------------------------------------------------
st.subheader("What each model costs to run")
cost = load_csv("cost.csv")
if cost is not None:
    rows = "".join(
        f'<tr><td><span class="dot" style="background:{MODEL_COLOR[r.model]}"></span> '
        f'{MODEL_LETTER[r.model]}: {MODEL_NAME[r.model]}</td><td>{r.size_mb:,.0f} MB</td>'
        f'<td>{r.ms_per_review:.1f} ms ({r.hardware.replace(" (batch of whole test set)", "")})</td>'
        f'<td>{r.training}</td></tr>' for r in cost.itertuples())
    st.html('<table class="plain"><tr><th>Model</th><th>Size</th><th>Per review</th><th>Training</th></tr>'
            f'{rows}</table>')
