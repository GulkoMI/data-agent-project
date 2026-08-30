# Cross-domain Reviews — Data Card

## Intended use

Educational binary sentiment classification and data-pipeline evaluation.

## Data

- Rows: 598
- Sources: `{'amazon_polarity': 300, 'steam_reviews': 298}`
- Labels: `{'negative': 316, 'positive': 282}`
- Mode: Hugging Face dataset snapshots
- Fields: record_id, text, source, source_label, auto_label, confidence, human_label, final_label.

## Provenance and licensing

Both configured sources are loaded through Hugging Face Datasets: `mteb/amazon_polarity` and [`reapxdev/steam-reviews-scraper`](https://huggingface.co/datasets/reapxdev/steam-reviews-scraper).
For Steam, `reviewText` supplies the text, `votedUp` supplies the negative/positive weak label, and `language` filters English rows. No Steam API or Steam token is used. Consult current dataset terms before redistributing raw text.

## Limitations

Ratings are imperfect sentiment labels. Reviews may contain personal information, abuse, sarcasm, language mismatch, and domain-specific bias.
