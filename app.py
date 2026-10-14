"""Streamlit app: Arabic/English customer-feedback sentiment.

    streamlit run app.py        ->  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Guest feedback sentiment", page_icon="💬", layout="wide")

from app_lib.ui import setup  # noqa: E402  (set_page_config must run first)

setup()
pages = [
    st.Page("app_pages/try_it.py", title="Try it", url_path="try", default=True),
    st.Page("app_pages/results.py", title="Results", url_path="results"),
    st.Page("app_pages/mistakes.py", title="Mistakes", url_path="mistakes"),
    st.Page("app_pages/data_story.py", title="The data", url_path="data"),
    st.Page("app_pages/batch.py", title="Batch", url_path="batch"),
]
st.navigation(pages, position="top").run()
