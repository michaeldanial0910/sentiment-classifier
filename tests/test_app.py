"""Smoke test: every page of the Streamlit app renders without an exception.

    pytest -q tests/test_app.py
"""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["../app_pages/try_it.py", "../app_pages/results.py", "../app_pages/mistakes.py",
         "../app_pages/data_story.py", "../app_pages/batch.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(page, default_timeout=120).run()
    assert not at.exception, at.exception
