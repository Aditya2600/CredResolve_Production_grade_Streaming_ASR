# Build Inventory

This script prepares an ASR data inventory from the source CSV, downloads each MP3, converts it to mono 16 kHz WAV, and writes the inventory in both CSV and Parquet formats.

## Input mapping

The normalized inventory fields are derived from the source CSV like this:

- `call_id <- call_sid`
- `audio_url <- cr_recording_url`
- `portfolio <- institute_name`
- `lender <- lender`
- `borrower_name <- name`
- `event_date <- due_date`
- `amount_1 <- principle`
- `amount_2 <- emi_amount`
- `amount_3 <- total_due`

All original source columns are preserved in the final inventory output after the normalized columns.

## Outputs

Running the script creates:

- `data/raw/{call_id}.mp3`
- `data/wav/{call_id}.wav`
- `outputs/data_inventory.csv`
- `outputs/data_inventory.parquet`

The inventory also includes pipeline bookkeeping columns for transcript, download, conversion, segmentation, alignment, quality tier, split, and slice tags. Audio segmentation is intentionally not performed in this step, so `segmentation_status` is set to `not_started`.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the builder:

```bash
python build_inventory.py --input query_result_2026-04-07T05_59_50.851541136Z.csv
```

Useful options:

```bash
python build_inventory.py \
  --input query_result_2026-04-07T05_59_50.851541136Z.csv \
  --retries 5 \
  --timeout 120 \
  --overwrite \
  --log-level DEBUG
```

## Notes

- `ffmpeg` must be installed and available on `PATH`.
- Downloads are retried with backoff and failed partial files are removed before retrying.
- Existing MP3 and WAV files are reused unless `--overwrite` is passed.
- Empty transcript fields are initialized but left blank for later ASR processing stages.
