import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (INK, LANG_NAME, MODEL_COLOR, MODEL_LETTER, MODEL_NAME, SENT, altair_theme, use_altair_theme, dot, esc,
                        load_csv, load_test_predictions, need_data, sentiment_chip, slip)
from src.config import LABELS

use_altair_theme()

st.title("Where the models go wrong, and why")
st.html('<p class="lede">I read 50 random mistakes from the best model (25 Arabic, 25 English) and wrote down why '
        'each one happened. Most aren\'t blunders: the review mixes praise with a complaint, or the star rating '
        'disagrees with what the guest wrote.</p>')

audit = load_csv("error_audit_sample.csv")
if audit is not None:
    order = ["adjacent class on a mixed review", "label contradicts text",
             "label debatable (text leans to predicted class)", "polarity flip (neg<->pos)"]
    nice = {"adjacent class on a mixed review": "One class off on a mixed review",
            "label contradicts text": "Stars contradict the text",
            "label debatable (text leans to predicted class)": "Stars debatable",
            "polarity flip (neg<->pos)": "Negative and positive swapped"}
    counts = audit.groupby(["category", "lang"]).size().rename("n").reset_index()
    counts["Reason"] = counts["category"].map(nice)
    counts["Language"] = counts["lang"].map(LANG_NAME)
    bar = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, height=24, stroke="#F7F8FA", strokeWidth=2).encode(
        y=alt.Y("Reason:N", sort=[nice[o] for o in order], title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=13, labelColor=INK)),
        x=alt.X("sum(n):Q", title="mistakes out of 50", axis=alt.Axis(tickMinStep=5)),
        color=alt.Color("Language:N", scale=alt.Scale(domain=["Arabic", "English"], range=["#24305E", "#9AA6C8"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Reason", "Language", alt.Tooltip("n:Q", title="mistakes")])
    st.altair_chart(bar.properties(height=210), theme=None, width="stretch")

    st.subheader("Six that explain the pattern")
    curated = [
        ("neutral", "negative", 0.99, "سئ للغايه. لايوجد شي. اللوبي صغير وريحة الممرات جدا كريهة… لاانصح بالسكن فيه",
         "“Extremely bad. Nothing [liked]. Small lobby, the corridors smell awful… I don't recommend staying here.” "
         "Three stars, clearly negative words: the label is wrong, not the model."),
        ("negative", "positive", 0.99, "رائع جداً ولي رجعة له بإذن الله. كل شيء جميل والأجمل الإفطار والنظافة… ضعف الإضاءة",
         "“Wonderful, and I'll come back. Everything was beautiful… weak lighting.” Rated 1–2 stars. Again the "
         "stars and the text disagree."),
        ("positive", "neutral", 0.86, "جيد. النظافة. الغرف ضيقة جداً",
         "“Good. Cleanliness. Rooms very cramped.” Booking's form asks what you liked and what you didn't, so one "
         "point each way reads as neutral."),
        ("positive", "neutral", 0.82, "pleasant stayed nights recently business, got good impression hotel breakfast "
         "buffet restaurant, waitress putting smile greeting morning…",
         "Mild praise. Where 'pleasant' ends and 'positive' begins is fuzzy even for people."),
        ("negative", "positive", 0.74, "soooo wanted love hotel did n't hotel going far removed mess waikiki…",
         "“Wanted to love it, didn't.” The contrast flips the meaning and the model misses it."),
        ("negative", "neutral", 0.62, "انقطاع الكرباء لساعات طويله على مدى يومين",
         "“Power cut for hours over two days.” A short fragment with no explicit sentiment word."),
    ]
    for true, pred, conf, text, why in curated:
        meta = (f'stars say {sentiment_chip(true)} model C said {sentiment_chip(pred)} {conf:.0%} sure'
                f'<span style="flex-basis:100%;color:{INK};font-size:0.92rem;margin-top:4px">{esc(why)}</span>')
        st.html(slip(text, meta))

# ---- explorer ----------------------------------------------------------------------------
st.subheader("Browse every test mistake")
test = load_test_predictions()
if test is None:
    need_data()
    st.stop()

c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
model = c1.selectbox("Model", list(MODEL_NAME), index=2, format_func=lambda k: f"{MODEL_LETTER[k]}: {MODEL_NAME[k]}")
lang = c2.selectbox("Language", ["all", "ar", "en"], format_func=LANG_NAME.get)
true = c3.selectbox("Stars say", ["any"] + LABELS)
pred = c4.selectbox("Model said", ["any"] + LABELS)
c5, c6 = st.columns([1.4, 2])
sort = c5.radio("Order", ["Most confident first", "Least confident first", "Random"], horizontal=True)
only_all_wrong = c6.toggle("Only reviews all three models got wrong")

wrong = test[test[f"{model}_pred"] != test["label"]]
if lang != "all":
    wrong = wrong[wrong["lang"] == lang]
if true != "any":
    wrong = wrong[wrong["label"] == true]
if pred != "any":
    wrong = wrong[wrong[f"{model}_pred"] == pred]
if only_all_wrong:
    wrong = wrong[(wrong[[f"{k}_pred" for k in MODEL_NAME]].ne(wrong["label"], axis=0)).all(axis=1)]
if sort == "Random":
    wrong = wrong.sample(frac=1, random_state=7)
else:
    wrong = wrong.sort_values(f"{model}_conf", ascending=(sort == "Least confident first"))

total = (test[f"{model}_pred"] != test["label"]).sum()
st.html(f'<p class="muted">{len(wrong):,} matching mistakes ({total:,} in total for this model, '
        f'out of {len(test):,} test reviews). Showing up to 12.</p>')
for r in wrong.head(12).itertuples():
    others = " ".join(
        f'<span class="chip">{dot(MODEL_COLOR[k])}{MODEL_LETTER[k]}: '
        f'{dot(SENT[getattr(r, f"{k}_pred")])}{getattr(r, f"{k}_pred")}</span>'
        for k in MODEL_NAME)
    meta = (f'{LANG_NAME[r.lang]} &nbsp; stars say {sentiment_chip(r.label)} '
            f'{MODEL_LETTER[model]} said {sentiment_chip(getattr(r, f"{model}_pred"))} '
            f'{getattr(r, f"{model}_conf"):.0%} sure &nbsp; {others}')
    text = r.text if len(r.text) < 700 else r.text[:700] + "…"
    st.html(slip(text, meta))
