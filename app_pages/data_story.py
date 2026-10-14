import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (INK, LANG_NAME, SENT, altair_theme, use_altair_theme, esc, figures, load_csv, load_reviews, need_data)
from src.config import LABELS

use_altair_theme()

st.title("Two sources of hotel feedback, and a word that gave the answer away")
st.html('<p class="lede">Arabic reviews come from Booking.com (HARD dataset, 2016), English ones from TripAdvisor. '
        'Both are hotel stays rated 1 to 5 stars, so one rule turns stars into labels for both languages: '
        '1–2 negative, 3 neutral, 4–5 positive.</p>')

leak = load_csv("leakage_words.csv")
st.subheader("The leak")
st.html('<div class="callout">When a Booking.com guest leaves no title, the site writes one from the score. '
        '<b>44% of Arabic reviews start with one of these 11 words, and every word belongs to exactly one star '
        'rating.</b> Left in, any model learns “this word means this label” and looks close to perfect on almost '
        'half the data without learning anything about sentiment. The word is stripped before training; titles '
        'the guest wrote themselves (in quotes) are kept.</div>')
if leak is not None:
    stars = {5: "positive", 4: "positive", 3: "neutral", 2: "negative", 1: "negative"}
    rows = "".join(
        f'<tr><td dir="rtl" style="font-size:1.05rem">{esc(r.word)}</td><td>{esc(r.meaning)}</td>'
        f'<td>{r.reviews:,}</td><td><span class="dot" style="background:{SENT[stars[r.stars]]}"></span> '
        f'{"★" * r.stars}</td><td>{r.share_with_that_rating:.0%}</td></tr>' for r in leak.itertuples())
    st.html('<table class="plain"><tr><th>Word</th><th>Means</th><th>Reviews</th><th>Always this rating</th>'
            f'<th>Share</th></tr>{rows}</table>')

reviews = load_reviews()
if reviews is None:
    need_data()
    st.stop()

st.subheader("What was used")
ar, en = reviews[reviews["lang"] == "ar"], reviews[reviews["lang"] == "en"]
st.html(figures([
    (f"{len(reviews):,}", "reviews after cleaning and removing duplicates"),
    (f"{len(ar):,}", "Arabic, a stratified sample of 409k"),
    (f"{len(en):,}", "English, all of them"),
    ("70 / 15 / 15", "train, validation and test, stratified by language and label"),
]))

c1, c2 = st.columns(2, gap="large")
mix = reviews.groupby(["lang", "label"]).size().rename("n").reset_index()
mix["share"] = mix["n"] / mix.groupby("lang")["n"].transform("sum")
mix["Language"] = mix["lang"].map(LANG_NAME)
c1.markdown("**About 70% of reviews are positive in both languages**")
bars = alt.Chart(mix).mark_bar(height=26).encode(
    y=alt.Y("Language:N", title=None),
    x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
    color=alt.Color("label:N", scale=alt.Scale(domain=LABELS, range=[SENT[l] for l in LABELS]),
                    legend=alt.Legend(orient="top", title=None)),
    order=alt.Order("label_order:Q"),
    tooltip=["Language", "label", alt.Tooltip("n:Q", title="reviews", format=","),
             alt.Tooltip("share:Q", format=".1%")],
).transform_calculate(label_order="indexof(['negative','neutral','positive'], datum.label)")
gaps = bars.mark_bar(height=26, stroke="#F7F8FA", strokeWidth=2)
c1.altair_chart(gaps.properties(height=130), theme=None, width="stretch")
c1.html('<p class="muted">That is why the metric is macro-F1: a model that always says “positive” would score '
        '70% accuracy and be useless.</p>')

lengths = reviews.assign(words=reviews["text"].str.split().str.len().clip(upper=300))
lengths["Language"] = lengths["lang"].map(LANG_NAME)
c2.markdown("**Arabic reviews are much shorter**")
hist = alt.Chart(lengths).mark_bar(opacity=0.85, binSpacing=1).encode(
    x=alt.X("words:Q", bin=alt.Bin(step=10), title="words per review (capped at 300)"),
    y=alt.Y("count():Q", title="reviews", stack=None),
    color=alt.Color("Language:N", scale=alt.Scale(domain=["Arabic", "English"], range=["#24305E", "#9AA6C8"]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["Language", alt.Tooltip("count():Q", title="reviews")])
c2.altair_chart(hist.properties(height=200), theme=None, width="stretch")
c2.html(f'<p class="muted">Median {int(ar["text"].str.split().str.len().median())} words in Arabic, '
        f'{int(en["text"].str.split().str.len().median())} in English. Booking\'s form also splits each review into '
        f'“liked” and “disliked”, so most Arabic reviews carry praise and a complaint at once.</p>')

st.subheader("Sources")
st.markdown(
    "- **HARD**: Elnagar, Khalifa & Einea (2018), *Hotel Arabic-Reviews Dataset Construction for Sentiment Analysis "
    "Applications*. [github.com/elnagara/HARD-Arabic-Dataset](https://github.com/elnagara/HARD-Arabic-Dataset)\n"
    "- **TripAdvisor hotel reviews** (20k), Alam, Ryu & Lee (2016). The text arrives lower-cased with many stop words "
    "removed, which suits TF-IDF better than transformers.\n"
    "- **XLM-R sentiment checkpoint**: Barbieri et al. (2022), *XLM-T*, `cardiffnlp/twitter-xlm-roberta-base-sentiment`.")
