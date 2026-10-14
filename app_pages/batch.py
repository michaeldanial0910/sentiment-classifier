import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import SENT, altair_theme, use_altair_theme, figures
from src.config import LABELS
from src.predict import predict_tfidf

use_altair_theme()

SAMPLE = """الغرفة نظيفة والموظفين محترمين جدا
the shower was cold every morning and nobody fixed it
موقع ممتاز لكن الإفطار عادي
great value, would stay again
الخدمة سيئة والانتظار طويل عند الاستقبال
room was fine, nothing to write home about"""

st.title("Sort a whole inbox at once")
st.html('<p class="lede">Paste one review per line, or upload a CSV with a <b>text</b> column. Arabic and English '
        'can be mixed. Negative reviews come first, and anything the model isn\'t sure about is flagged for a '
        'person. This uses the fast TF-IDF model.</p>')

c1, c2 = st.columns([1.3, 1], gap="large")
pasted = c1.text_area("One review per line", value=SAMPLE, height=200)
upload = c2.file_uploader("Or upload a CSV", type="csv")

texts = []
if upload is not None:
    data = pd.read_csv(upload)
    if "text" not in data.columns:
        st.error("The CSV needs a column named “text”. Rename the review column and upload it again.")
        st.stop()
    texts = data["text"].dropna().astype(str).tolist()
else:
    texts = [t.strip() for t in pasted.splitlines() if t.strip()]

if not texts:
    st.info("Add at least one review to sort.")
    st.stop()

out = predict_tfidf(texts)
out["priority"] = out["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
out = out.sort_values(["priority", "confidence"], ascending=[True, False]).drop(columns=["priority", "model"])

st.html(figures([
    (f"{len(out):,}", "reviews sorted"),
    (f"{(out['sentiment'] == 'negative').mean():.0%}", "negative, shown first"),
    (f"{int(out['needs_review'].sum()):,}", "flagged for a person (low confidence)"),
]))

share = out["sentiment"].value_counts(normalize=True).reindex(LABELS, fill_value=0).rename("share").reset_index()
share["order"] = range(3)
strip = alt.Chart(share).mark_bar(height=22, stroke="#F7F8FA", strokeWidth=2).encode(
    x=alt.X("share:Q", stack="normalize", axis=None),
    color=alt.Color("sentiment:N", scale=alt.Scale(domain=LABELS, range=[SENT[l] for l in LABELS]),
                    legend=alt.Legend(orient="top", title=None)),
    order="order:Q", tooltip=["sentiment", alt.Tooltip("share:Q", format=".0%")])
st.altair_chart(strip.properties(height=60), theme=None, width="stretch")

out["confidence"] = (out["confidence"] * 100).round()
show = out.rename(columns={"text": "Review", "lang": "Language", "sentiment": "Sentiment",
                           "confidence": "Confidence", "needs_review": "Send to a person"})
st.dataframe(
    show[["Review", "Sentiment", "Confidence", "Send to a person", "Language"] + [f"p_{l}" for l in LABELS]],
    width="stretch", hide_index=True, height=min(420, 38 + 35 * len(show)),
    column_config={"Confidence": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
                   **{f"p_{l}": st.column_config.NumberColumn(f"P({l})", format="%.2f") for l in LABELS}})
st.download_button("Download sorted reviews (CSV)", out.assign(confidence=out["confidence"] / 100).to_csv(index=False).encode("utf-8"),
                   "sorted_reviews.csv", "text/csv")
