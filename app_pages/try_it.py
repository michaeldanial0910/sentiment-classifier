import streamlit as st

from app_lib.ui import (MODEL_COLOR, MODEL_NAME, SENT, dot, esc, load_test_predictions, sentiment_chip, slip,
                        verdict_card)
from src.predict import available_transformer_name, predict_tfidf, predict_transformer

EXAMPLES = {
    "Mixed, Arabic": "الفطور ممتاز والموظفين متعاونين، لكن الغرفة صغيرة جدا والسعر مبالغ فيه",
    "Complaint, Arabic": "لن اكرر التجربة، الغرفة متسخة والموظفين غير متعاونين ابدا",
    "Praise, Arabic": "الموقع رائع وقريب من الحرم والنظافة ممتازة وطاقم الاستقبال بشوشين",
    "Contrast, English": "so wanted to love this hotel but the room was tired and the staff couldn't care less",
    "Lukewarm, English": "hotel was ok, nothing special. breakfast average, location fine for one night",
}

st.title("Read a guest review, in Arabic or English")
st.html('<p class="lede">Paste any hotel review, or pull a real one from the test set. The model sorts it into '
        'negative, neutral or positive, shows how sure it is, and flags the ones a person should read.</p>')

if "review" not in st.session_state:
    st.session_state.review = EXAMPLES["Mixed, Arabic"]
    st.session_state.test_row = None


def use_example():
    st.session_state.review = EXAMPLES[st.session_state.example_pick]
    st.session_state.test_row = None


def use_test_review():
    test = load_test_predictions()
    if test is None:
        return
    row = test.sample(1).iloc[0]
    st.session_state.example_pick = None
    st.session_state.review = row["text"]
    st.session_state.test_row = row.to_dict()


st.pills("Examples", list(EXAMPLES), key="example_pick", on_change=use_example, label_visibility="collapsed")
left, right = st.columns([1.15, 1], gap="large")

with left:
    text = st.text_area("Review", key="review", height=180, label_visibility="collapsed",
                        placeholder="اكتب تقييمك هنا… or type a review in English")
    c1, c2 = st.columns([1, 1])
    c1.button("Pull a random test review", on_click=use_test_review, width="stretch",
              disabled=load_test_predictions() is None)
    run_tf = c2.toggle("Also run the transformer", value=False,
                       help="Needs torch + transformers. Uses models/xlmr_finetuned/ if you downloaded it from "
                            "Google Drive, otherwise the public pretrained checkpoint (about 1.1 GB, first run only).")
    st.html('<p class="muted">The baseline runs instantly on a laptop CPU. The transformer is about 85 times larger '
            'and was 0.03 macro-F1 better on the test set.</p>')

with right:
    if text.strip():
        st.html(verdict_card(predict_tfidf([text]).iloc[0], "A: TF-IDF + logistic regression",
                             MODEL_COLOR["tfidf_lr"]))
        if run_tf:
            try:
                with st.spinner("Loading the transformer"):
                    out = predict_transformer([text]).iloc[0]
                key = "finetuned_xlmr" if out["model"].startswith("C") else "zeroshot_xlmr"
                st.html(verdict_card(out, out["model"].replace(" · ", ": "), MODEL_COLOR[key]))
            except Exception as e:  # torch missing, no network for the first download, ...
                if "torch" in str(e) or "transformers" in str(e):
                    st.warning("The transformer needs two extra packages. Install them with "
                               "`pip install -r requirements-transformer.txt`, then switch this on again.")
                else:
                    st.warning(f"The transformer couldn't run: {e}")
    else:
        st.html('<p class="muted">Type or paste a review on the left to see its sentiment.</p>')

row = st.session_state.test_row
if row is not None and text == row["text"]:
    st.subheader("What all three models said about this test review")
    st.html('<p class="muted">Stored predictions from the evaluation run, on the exact same review. '
            'The label comes from the guest\'s star rating.</p>')
    cells = []
    for key in MODEL_NAME:
        if f"{key}_pred" not in row:
            continue
        pred = row[f"{key}_pred"]
        ok = pred == row["label"]
        cells.append(f'<span class="chip">{dot(MODEL_COLOR[key])}{esc(MODEL_NAME[key])}: '
                     f'{dot(SENT[pred])}<b>{pred}</b> {row[f"{key}_conf"]:.0%}'
                     f'{"" if ok else " (wrong)"}</span>')
    meta = f'stars say {sentiment_chip(row["label"])} {"".join(cells)}'
    st.html(slip(row["text"], meta))
