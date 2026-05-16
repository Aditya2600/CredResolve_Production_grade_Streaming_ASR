# Punjabi (Gurmukhi) gold sets — native-speaker review block

**PR merge is blocked until this file is removed.** The corresponding
test, `tests/grammars/pa/test_review_block.py::test_no_punjabi_review_required_marker`,
fails while this file exists, so any CI configured to gate on tests
will refuse to merge.

## What this gates

The Punjabi gold sets in this directory were drafted by an automated
template-port from the Hindi / Marathi gold sets. Punjabi number
words, scale words, half / quarter compounds, month names, time cues,
currency cues, and percent cues are lexically distinct from
Hindi / Marathi, so the gold sets must be reviewed by a native
Punjabi speaker before they can serve as the truth source for grammar
acceptance.

## Special handling for this language

* **NFC + bindi-fold in the working copy.** Punjabi text is folded
  to its non-bindi (no-nukta) base letters in the working copy —
  see `runtime/unicode_clean.py::_fold_gurmukhi_bindi`. Five pairs
  are in scope:

  ```
  ਫ਼ -> ਫ      ਜ਼ -> ਜ      ਗ਼ -> ਗ      ਖ਼ -> ਖ      ਸ਼ -> ਸ
  ```

  All gold-set `raw` entries can use either form; the WFST sees the
  folded text. Gold-set `expected` entries are Latin digits, so the
  fold doesn't change them. Reviewers must confirm that the fold
  does not erase a phonemic contrast meaningful in the dialects they
  are sign-off authority for; flag any such case in this file.

* **Half / quarter compounds.** The Punjabi shapes are:

  ```
  ਸਵਾ  — 1.25 / 1¼   (Hindi: सवा)
  ਡੇਢ  — 1.5  / 1½   (Hindi: डेढ़)
  ਢਾਈ  — 2.5  / 2½   (Hindi: ढाई)
  ਸਾਢੇ — N + 0.5     (Hindi: साढ़े)
  ਪੌਣੇ — N − 0.25    (Hindi: पौने)
  ```

  Every compound family must appear in `cardinal.jsonl` and again
  in `time.jsonl` (the time grammar reuses the same modifiers for
  quarter-past / quarter-to / half-past).

## Reviewer checklist

For each file in this directory, the reviewer must verify:

* `cardinal.jsonl`   — every 0..99 entry, every hundred compound
  (ਇੱਕ ਸੌ … ਨੌਂ ਸੌ — note Punjabi prefers the multi-word shape
  ``ਇੱਕ ਸੌ`` rather than a single token; the gold set must lock
  this), every scale combination (ਹਜ਼ਾਰ / ਲੱਖ / ਕਰੋੜ — folded to
  ਹਜਾਰ / ਲੱਖ / ਕਰੋੜ in the working copy), every half/quarter
  compound family (ਸਵਾ / ਡੇਢ / ਢਾਈ / ਸਾਢੇ / ਪੌਣੇ), and every
  dialectal spelling alternate listed in the grammar.
* `currency.jsonl`   — ਰੁਪਏ / ਰੁਪਇਆ / ਰੁ. cue acceptance,
  ਪੈਸੇ / ਪੈਸਾ paise cue, Indian-grouped output (1,25,000). The
  bindi-fold means the grammar only ever sees ``ਜ`` for both ``ਜ``
  and ``ਜ਼`` — reviewers should confirm no Punjabi currency lemma
  depends on a bindi distinction that the fold would erase.
* `date.jsonl`       — Punjabi month names. Both the Punjabi calendar
  month names (e.g. ਚੇਤ, ਵੈਸਾਖ, ਜੇਠ) and the Gregorian Punjabi
  spellings (ਜਨਵਰੀ, ਫ਼ਰਵਰੀ -> ਫਰਵਰੀ post-fold, ਅਪ੍ਰੈਲ, ਜੁਲਾਈ,
  ਅਗਸਤ, ਸਤੰਬਰ, ਅਕਤੂਬਰ, ਨਵੰਬਰ, ਦਸੰਬਰ) must be exercised.
* `time.jsonl`       — ਵਜੇ / ਵਜਕੇ / ਮਿੰਟ cues, ਸਵੇਰੇ / ਦੁਪਹਿਰੇ /
  ਸ਼ਾਮ -> ਸਾਮ / ਰਾਤ time-of-day modifiers, half/quarter time
  compounds.
* `percent.jsonl`    — ਪ੍ਰਤੀਸ਼ਤ -> ਪ੍ਰਤੀਸਤ (post-fold) / ਫ਼ੀਸਦੀ ->
  ਫੀਸਦੀ (post-fold) cue acceptance.
* `phone.jsonl`      — Gurmukhi-digit passthrough (੦-੯,
  U+0A66..U+0A6F).
* `id.jsonl`         — PAN / Aadhaar / IFSC / policy-number forms.

The reviewer should annotate any entry they want to remove, add, or
correct in this file, then sign off below:

```
Reviewed by:    <name>
Date:           <YYYY-MM-DD>
Sign-off:       <native speaker / dialect — e.g., "Majhi",
                 "Doabi", "Malwai">
Decisions:      <free-form notes — which alternates rejected / added,
                 any bindi-fold false-merges flagged>
```

## Why the block

Punjabi is the third language onboarded under the bn/gu/pa template
stage. The structural shape is direct-copy from Hindi/Marathi but
two language-specific concerns must be reviewed:

1. The bindi-fold (`runtime/unicode_clean.py`) erases the
   ਫ਼/ਜ਼/ਗ਼/ਖ਼/ਸ਼ distinction in the working copy. The fold is
   conservative — only five pairs, LLA (ਲ਼) deliberately excluded —
   but the reviewer must confirm that no in-vocabulary lemma
   depends on a bindi distinction the fold collapses.
2. Punjabi shares many number words and scale words with Hindi at
   the codepoint level but diverges on half/quarter compounds and
   month names; the gold set must exercise the divergence.

## How to unblock

After review, delete this file (`git rm REVIEW_REQUIRED.md`) and the
gold-set tests will start running. CI will then enforce the per-class
acceptance bars defined in `tests/grammars/pa/test_*.py` (≥ 98 % per
file, ≥ 200 entries per file, every compound family present — same
contract as the Marathi template).
