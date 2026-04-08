# Context Biasing Assets

Put deployment-managed phrase files under `context_biasing/phrases/`.

Expected format:

- one file per language, for example `hi.txt`
- one underscore-delimited phrase group per line
- first token is the canonical term
- remaining tokens are accepted variants or spellings
- blank lines and lines starting with `#` are ignored

Example:

```text
loan id_loan id_लोन आईडी
cred resolve_credresolve_क्रेड रिजॉल्व
```

You can also generate a phrase file from seed terms plus repeated WER confusions:

```bash
python tools/build_phrase_lexicon.py \
  --errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --seed-terms-file artifacts/domain_terms_seed.txt \
  --out-phrases-file context_biasing/phrases/hi.txt \
  --out-review-json artifacts/hi_phrase_lexicon_review.json
```

Notes:

- Plain-text seed files may contain either one phrase per line or underscore-delimited groups.
- CSV and TSV seed files may contain `canonical` plus `variant` or `variants` columns.
- The generated phrase file is ready for inference-time context biasing, but review the JSON artifact before deployment so generic phrases do not get boosted accidentally.
