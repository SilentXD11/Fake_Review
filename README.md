# Fake Review Detection with BERT

Fine-tunes a pretrained BERT model to tell **deceptive (fake)** hotel reviews from **truthful** ones.

**Result: 85.9% accuracy on 320 held-out reviews.**

Everything is in one Jupyter notebook, `Fake_Review.ipynb`, with its outputs saved so the results can be checked without re-running it.

---

## Dataset

[Deceptive Opinion Spam Corpus](https://myleott.com/op-spam.html) (Ott et al.), included here as `deceptive-opinion.csv`.

- **1,600** reviews of 20 Chicago hotels
- **Balanced:** 800 truthful (from TripAdvisor and other sites) and 800 deceptive (written to order via Mechanical Turk)
- Positive and negative reviews in both classes

Because the classes are balanced, accuracy is a fair headline metric here.

## Method

| Step | Detail |
|---|---|
| Cleaning | Lowercase, remove URLs and non-letter characters, collapse whitespace |
| Split | 80% train (1,280) / 20% validation (320), `random_state=42` |
| Tokeniser | `bert-base-uncased`, max length 256 |
| Model | `BertForSequenceClassification`, 2 labels |
| Training | Hugging Face `Trainer`, 3 epochs, batch size 8, 500 warm-up steps, weight decay 0.01 |

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **85.94%** (275 / 320) |
| Final training loss | 0.414 |
| Training time | 21 s on a single GPU |

## Run it

```bash
pip install transformers torch scikit-learn pandas numpy accelerate
jupyter notebook Fake_Review.ipynb
```

Run the cells in order. The last cells load the trained model and let you type a review to classify.

## Limitations and next steps

- Only accuracy is reported. Next: precision, recall, F1 and a confusion matrix.
- Single train/validation split. Next: k-fold cross-validation for a more reliable estimate.
- All reviews are about hotels in one city, so results may not carry over to other domains.
- Short informal reviews typed by hand are often classed as fake; the model has learned the style of the corpus, not general deception.

## Tech

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas

## License

MIT
