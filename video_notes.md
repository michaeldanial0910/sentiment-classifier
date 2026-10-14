# Walkthrough video: notes (target 4–5 min)

Record the Streamlit app (`streamlit run app.py`, browser at full screen) instead of the code.

1. **Try it (60 s).** One sentence on the problem: a hotel group gets feedback in Arabic and English and wants
   complaints first. Click *Mixed, Arabic*: point at the probability strip and the "send to a person" flag.
   Click *Contrast, English*. Then *Pull a random test review* and show what all three models said about it,
   next to the star rating.
2. **The data (60 s).** The leakage table: 44% of Arabic reviews start with a Booking.com word that *is* the
   label, so it is stripped before training. Then the label mix (about 70% positive, which is why the metric
   is macro-F1) and the short Arabic reviews.
3. **Results (75 s).** The fair setup in one line: same split, same test rows, same evaluation code, and the
   Colab run checked the dataset's SHA-256 before training. The interval chart: fine-tuned XLM-R best (0.697),
   only +0.034 over TF-IDF, the untrained transformer worst (0.517). Switch the confusion matrix to Arabic:
   positive read as neutral is the main error. Move the triage slider: at 0.80 the model handles 72% of
   reviews at about 90% accuracy.
4. **Mistakes (45 s).** The audit bar: 70% are one class off on mixed reviews, about a quarter are star
   ratings that contradict the text. Read one example of each. Toggle "only reviews all three got wrong".
5. **Batch (20 s).** The pasted inbox: negatives first, flagged rows, the download button.
6. **Limitations and next steps (20 s).** Star ratings as labels, one domain, two sources that differ in more
   than language. Next: a human-labelled sample and aspect-based sentiment (room, staff, location, price).
