# Walkthrough video: notes (target 4–5 min)

1. **Problem (30 s).** A hotel group gets feedback in Arabic and English. Goal: sort it into negative / neutral / positive
   so the support team sees complaints first. Compare a classical model against transformers fairly.
2. **Data and the leakage catch (60 s).** Open `notebooks/01_eda.ipynb`.
   - Two sources, same domain (hotel reviews), same 1–5 star scale → same label rule.
   - Show the leakage table: 44% of Arabic reviews start with a Booking.com score word that *is* the label. Stripped.
   - Label mix: about 70% positive, so macro-F1 is the metric, not accuracy.
3. **Fair comparison setup (40 s).** One split, stratified by language × label. Same test rows for every model.
   The Colab notebook rebuilds the data and checks its SHA-256 against the local file before training.
   One evaluation script scores all three.
4. **Results (60 s).** Show `reports/figures/model_comparison.png` and the confusion matrices.
   Read the headline numbers from the README table; point at neutral as the hard class.
5. **Errors (45 s).** Two or three examples from the README error table: a mixed "liked/disliked" review,
   a review whose stars contradict its text, a short Arabic fragment.
6. **Demo (45 s).** `streamlit run app.py`: an Arabic and an English example, the low-confidence flag,
   the batch tab with a CSV. CLI: `python -m src.predict "..."`.
7. **Limitations and what I'd do next (30 s).** Simulated vs real feedback, noisy star labels, one domain,
   pre-processed English text. Next: human-labelled sample, aspect-based sentiment (room/staff/location).
