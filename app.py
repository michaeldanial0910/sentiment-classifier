"""Streamlit demo: Arabic/English customer-feedback sentiment.

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.config import FIGURES, LABELS, REPORTS
from src.predict import available_transformer_name, predict_tfidf, predict_transformer

st.set_page_config(page_title="Feedback Sentiment (AR/EN)", page_icon="💬", layout="wide")

EXAMPLES = {
    "English · mixed": "great location and friendly staff, but the room was dirty and the wifi never worked",
    "English · neutral": "hotel was ok, nothing special, breakfast average, would probably try somewhere else next time",
    "Arabic · positive": "الموقع ممتاز والموظفين متعاونين جدا والغرف نظيفة وواسعة",
    "Arabic · negative": "الخدمة سيئة جدا والغرفة غير نظيفة ولن أكرر التجربة",
    "Arabic · mixed": "الفطور جيد لكن الغرفة صغيرة والسعر مرتفع مقارنة بالخدمة",
}
COLORS = {"negative": "#c0504d", "neutral": "#9e9e9e", "positive": "#4f81bd"}

st.title("💬 Customer feedback sentiment: Arabic & English")
st.caption("Hotel-review models from the UneeQ ML track. Predictions are suggestions for a support team, "
           "not automatic decisions; low-confidence ones are flagged for a human.")

use_tf = st.sidebar.checkbox(f"Also run the transformer ({available_transformer_name()})", value=False,
                             help="Needs torch + transformers installed. The first run downloads/loads ~1.1 GB.")
tab_one, tab_batch, tab_results = st.tabs(["Single review", "Batch (CSV)", "Model comparison"])


def show(df: pd.DataFrame):
    r = df.iloc[0]
    st.markdown(f"**{r['model']}** → <span style='color:{COLORS[r['sentiment']]};font-weight:700'>"
                f"{r['sentiment'].upper()}</span> ({r['confidence']:.0%}, detected language: {r['lang']})",
                unsafe_allow_html=True)
    st.bar_chart(pd.DataFrame({"probability": [r[f"p_{l}"] for l in LABELS]}, index=LABELS), height=180)
    if r["needs_review"]:
        st.warning("Low confidence: route this one to a person.")


with tab_one:
    pick = st.selectbox("Try an example", ["(type your own)"] + list(EXAMPLES))
    text = st.text_area("Review", value=EXAMPLES.get(pick, ""), height=110)
    if st.button("Classify", type="primary") and text.strip():
        cols = st.columns(2 if use_tf else 1)
        with cols[0]:
            show(predict_tfidf([text]))
        if use_tf:
            with cols[1]:
                try:
                    with st.spinner("Loading transformer…"):
                        show(predict_transformer([text]))
                except Exception as e:  # missing torch, no internet for the first download, ...
                    st.error(f"Transformer unavailable: {e}")

with tab_batch:
    st.write("Upload a CSV with a **text** column (Arabic, English or mixed). Uses the TF-IDF model.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        if "text" not in data.columns:
            st.error("The file needs a column called 'text'.")
        else:
            out = predict_tfidf(data["text"].astype(str).tolist())
            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews", len(out))
            c2.metric("Negative", f"{(out['sentiment'] == 'negative').mean():.0%}")
            c3.metric("Flagged for review", int(out["needs_review"].sum()))
            st.dataframe(out.drop(columns=["model"]), width="stretch", height=360)
            st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab_results:
    metrics = REPORTS / "metrics.csv"
    if metrics.exists():
        m = pd.read_csv(metrics)
        table = m.pivot_table(index="name", columns="scope", values="macro_f1").reindex(columns=["all", "ar", "en"])
        table.columns = ["Both", "Arabic", "English"]
        st.subheader("Test macro-F1 (same 6,074 test reviews for every model)")
        st.dataframe(table.style.format("{:.3f}"), width="stretch")
    for fig, caption in (("model_comparison.png", "Scores with 95% bootstrap intervals"),
                         ("confusion_matrices.png", "Where each model goes wrong")):
        if (FIGURES / fig).exists():
            st.image(str(FIGURES / fig), caption=caption)
