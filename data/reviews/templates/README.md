# Partner / Manual Export Format

If a partner brand shares review data with you, or if you export from any
external system, the format expected by `LocalFileScraper` is below.

## Required columns

| column | type | required | notes |
|---|---|---|---|
| `text` | str | ✅ | The review body |
| `source` | str | recommended | Where it came from; e.g. `partner_喜茶`, `manual_xhs`, `survey_2026_q2` |
| `brand` | str | recommended | Brand name (Chinese) |
| `sku` | str | optional | Product name |
| `customization_raw` | str | optional | "三分糖少冰加芋圆" — preserved verbatim |
| `rating` | float | optional | 1.0–5.0 if available |
| `user_id_external` | str | optional | **Will be hashed** before persistence |
| `source_url` | str | optional | Permalink |
| `timestamp` | str | optional | ISO 8601 if known |

## Format flexibility

The ingester (`scripts/ingest_partner.py`) accepts:
- `.csv` (UTF-8, comma-separated)
- `.tsv`
- `.jsonl` (one JSON object per line)
- `.json` (single JSON array)

## Privacy / compliance

- Real user identifiers (`user_id_external`) are SHA256-hashed via
  `scrapers.base.hash_user_id` before they ever touch the parquet store.
- Source URLs are kept verbatim — review before publishing data; for sensitive
  cases set them to null.
- `partner_export_template.csv` shows the field layout. Replace the example
  rows with your data.

## Ingestion

```bash
python scripts/ingest_partner.py \
    --path data/inbox/2026_q2_partner_export.csv \
    --shard partner_q2 \
    --source-tag partner_brandX
```

After ingest, files in `data/inbox/` should be moved to a private archive;
do **NOT** commit raw partner data to the repo.
