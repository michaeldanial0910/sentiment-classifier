"""Shared look, data loading and small HTML components for the Streamlit app."""
import html
import json

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from src.config import LABELS, PREDICTIONS, PROCESSED, REPORTS

# ---- design tokens -------------------------------------------------------------------
INK = "#1C2333"
INK_SOFT = "#5A6478"
RULE = "#E3E6EC"
PAPER = "#F7F8FA"
# Sentiment is diverging: two poles + a gray midpoint (validated CVD-safe: ΔE 25 protan).
SENT = {"negative": "#D4562E", "neutral": "#8A94A6", "positive": "#2A6FD6"}
SENT_AR = {"negative": "سلبي", "neutral": "محايد", "positive": "إيجابي"}
# Models are categorical identity (validated all-pairs: CVD and normal-vision pass).
MODEL_COLOR = {"tfidf_lr": "#6250D6", "zeroshot_xlmr": "#B07D00", "finetuned_xlmr": "#13976A"}
MODEL_NAME = {"tfidf_lr": "TF-IDF + LogReg", "zeroshot_xlmr": "XLM-R, no training",
              "finetuned_xlmr": "XLM-R fine-tuned"}
MODEL_LETTER = {"tfidf_lr": "A", "zeroshot_xlmr": "B", "finetuned_xlmr": "C"}
LANG_NAME = {"ar": "Arabic", "en": "English", "all": "Both languages"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, td, th, textarea, input, select,
button p, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{
  font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 4.2rem; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.35rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.45rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.1rem; }}
p, li {{ line-height: 1.6; }}
.lede {{ color: {INK_SOFT}; font-size: 1.08rem; max-width: 70ch; margin-top: -0.4rem; }}
.muted {{ color: {INK_SOFT}; font-size: 0.92rem; }}
.stTextArea textarea {{ font-size: 1.12rem; line-height: 1.7; unicode-bidi: plaintext; }}

/* review "slip": one guest review, rendered in its own script direction */
.slip {{ background: #fff; border: 1px solid {RULE}; border-radius: 10px; padding: 16px 18px; margin: 6px 0 12px; }}
.slip .text {{ font-size: 1.04rem; line-height: 1.75; unicode-bidi: plaintext; white-space: pre-wrap; }}
.slip .meta {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; font-size: 0.85rem; color: {INK_SOFT}; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 2px 10px;
         font-size: 0.82rem; font-weight: 500; border: 1px solid {RULE}; background: #fff; color: {INK}; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.wrong {{ text-decoration: line-through; text-decoration-color: {INK_SOFT}; }}

/* the sentiment strip: probabilities as one bar split into three */
.verdict {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }}
.verdict .head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.verdict .model {{ font-size: 0.9rem; color: {INK_SOFT}; }}
.verdict .label {{ font-size: 1.7rem; font-weight: 600; line-height: 1.1; }}
.verdict .swatch {{ display: inline-block; width: 18px; height: 18px; border-radius: 5px; margin-inline-end: 10px; vertical-align: -1px; }}
.verdict .label small {{ font-size: 1rem; font-weight: 400; color: {INK_SOFT}; margin-inline-start: 8px; }}
.strip {{ display: flex; height: 14px; border-radius: 7px; overflow: hidden; gap: 2px; margin: 14px 0 8px; background: #fff; }}
.strip span {{ display: block; height: 100%; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.84rem; color: {INK_SOFT}; font-variant-numeric: tabular-nums; }}
.flag {{ margin-top: 10px; font-size: 0.88rem; padding: 8px 12px; border-radius: 8px; background: #FFF6E5; color: #6B4B00; }}

/* figures in prose */
.figure-row {{ display: flex; gap: 28px; flex-wrap: wrap; margin: 10px 0 4px; }}
.figure {{ min-width: 150px; }}
.figure .n {{ font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.figure .l {{ font-size: 0.88rem; color: {INK_SOFT}; max-width: 22ch; }}
.callout {{ border-inline-start: 4px solid {INK}; background: #fff; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 12px 0; }}
table.plain {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.95rem; }}
table.plain th {{ text-align: start; font-weight: 500; color: {INK_SOFT}; border-bottom: 1px solid {RULE}; padding: 8px 10px; }}
table.plain td {{ border-bottom: 1px solid {RULE}; padding: 8px 10px; }}
table.plain td.best {{ font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
@media (max-width: 640px) {{ h1 {{ font-size: 1.8rem; }} .figure .n {{ font-size: 1.6rem; }} }}
</style>
"""


def setup():
    st.html(CSS)


def esc(s) -> str:
    return html.escape(str(s))


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def sentiment_chip(label: str, strike=False) -> str:
    cls = "chip wrong" if strike else "chip"
    return f'<span class="{cls}">{dot(SENT[label])}{label}</span>'


def model_chip(key: str) -> str:
    return f'<span class="chip">{dot(MODEL_COLOR[key])}{MODEL_LETTER[key]}: {MODEL_NAME[key]}</span>'


def verdict_card(row: pd.Series, model_label: str, color: str) -> str:
    probs = {l: float(row[f"p_{l}"]) for l in LABELS}
    label = row["sentiment"]
    segs = "".join(f'<span style="width:{max(probs[l], 0.004) * 100:.1f}%;background:{SENT[l]}" '
                   f'title="{l} {probs[l]:.0%}"></span>' for l in LABELS)
    legend = "".join(f'<span>{dot(SENT[l])} {l} {probs[l]:.0%}</span>' for l in LABELS)
    flag = ('<div class="flag">Low confidence for this model. In the support queue this review '
            'would go to a person instead of being labelled automatically.</div>') if row["needs_review"] else ""
    return f"""
    <div class="verdict">
      <div class="head">
        <span class="model">{dot(color)} {esc(model_label)}</span>
        <span class="model">detected: {LANG_NAME[row['lang']]}</span>
      </div>
      <div class="label"><span class="swatch" style="background:{SENT[label]}"></span>{label}<small>{probs[label]:.0%} confident</small></div>
      <div class="strip">{segs}</div>
      <div class="legend">{legend}</div>
      {flag}
    </div>"""


def slip(text: str, meta_html: str = "") -> str:
    return f'<div class="slip"><div class="text" dir="auto">{esc(text)}</div><div class="meta">{meta_html}</div></div>'


def figures(items) -> str:
    cells = "".join(f'<div class="figure"><div class="n">{esc(n)}</div><div class="l">{esc(l)}</div></div>'
                    for n, l in items)
    return f'<div class="figure-row">{cells}</div>'


# ---- data ----------------------------------------------------------------------------
@st.cache_data
def load_reviews() -> pd.DataFrame | None:
    path = PROCESSED / "reviews.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_test_predictions() -> pd.DataFrame | None:
    """One row per test review with gold label and every model's prediction + confidence."""
    reviews = load_reviews()
    if reviews is None:
        return None
    test = reviews[reviews["split"] == "test"][["id", "lang", "label", "text"]].set_index("id")
    for key in MODEL_NAME:
        path = PREDICTIONS / f"{key}_test.csv"
        if not path.exists():
            continue
        p = pd.read_csv(path).set_index("id")
        test[f"{key}_pred"] = p["pred"]
        test[f"{key}_conf"] = p[[f"p_{l}" for l in LABELS]].max(axis=1)
        for l in LABELS:
            test[f"{key}_p_{l}"] = p[f"p_{l}"]
    return test.reset_index()


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def need_data():
    st.info("This view needs the processed dataset. Run `python -m src.data` once (see README, *How to run*).")


def altair_theme():
    """Recessive axes and grid, Plex font, ink text: applied to every chart."""
    return {
        "config": {
            "font": "IBM Plex Sans Arabic, system-ui, -apple-system, Helvetica, Arial, sans-serif",
            "background": "transparent",
            "view": {"stroke": None},
            "axis": {"labelColor": INK_SOFT, "titleColor": INK_SOFT, "gridColor": RULE, "domainColor": RULE,
                     "tickColor": RULE, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500},
            "legend": {"labelColor": INK, "titleColor": INK_SOFT, "labelFontSize": 12, "titleFontWeight": 500},
            "text": {"font": "IBM Plex Sans Arabic, system-ui, -apple-system, Helvetica, Arial, sans-serif"},
            "header": {"labelColor": INK, "labelFontSize": 13, "labelFontWeight": 500, "titleColor": INK_SOFT},
            "title": {"color": INK, "fontWeight": 600, "fontSize": 14, "anchor": "start"},
        }
    }


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def use_altair_theme():
    alt.theme.register("plex", enable=True)(altair_theme)
