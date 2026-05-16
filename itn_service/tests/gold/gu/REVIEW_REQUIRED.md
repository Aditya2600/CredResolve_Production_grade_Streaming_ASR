# Gujarati gold sets — native-speaker review block

**PR merge is blocked until this file is removed.** The corresponding
test, `tests/grammars/gu/test_review_block.py::test_no_gujarati_review_required_marker`,
fails while this file exists, so any CI configured to gate on tests
will refuse to merge.

## What this gates

The Gujarati gold sets in this directory were drafted by an automated
template-port from the Hindi / Marathi gold sets. Gujarati number
words, scale words (લાખ, કરોડ), half / quarter compounds, month names,
time cues, currency cues, and percent cues are lexically distinct
from Hindi and Marathi even though all three share Indian (lakh /
crore) grouping, so the gold sets must be reviewed by a native
Gujarati speaker before they can serve as the truth source for
grammar acceptance.

## Special handling for this language

* **Indian grouping is universal.** Gujarati commerce text always
  uses 1,25,000-style grouping (lakh / crore) regardless of the
  numbering system the digits are rendered in. The money grammar's
  output for ``એક લાખ પચીસ હજાર`` is ``₹1,25,000`` — never
  ``₹125,000``. The gold set must include several lakh / crore
  cases to exercise this.
* **Lakh / crore vocabulary.** ``લાખ`` (lakh) and ``કરોડ`` (crore)
  are the standard forms; ``અબજ`` (arab / billion) is acceptance-
  only. Reviewers should confirm the spelling choice for each.

## Reviewer checklist

For each file in this directory, the reviewer must verify:

* `cardinal.jsonl`   — every 0..99 entry, every hundred compound
  (એક સો … નવ સો), every scale combination (હજાર / લાખ / કરોડ),
  half/quarter compound family (દોઢ / અઢી / સવા / સાડા / પોણા),
  and every dialectal spelling alternate listed in the grammar.
* `currency.jsonl`   — રૂપિયા / રૂપિયો / રૂ. cue acceptance,
  પૈસા / પૈસો paise cue, Indian-grouped output (1,25,000).
* `date.jsonl`       — Gujarati month names (especially જાન્યુઆરી,
  ફેબ્રુઆરી, એપ્રિલ, જુલાઈ, ઓગસ્ટ, સપ્ટેમ્બર, ઓક્ટોબર, નવેમ્બર,
  ડિસેમ્બર). Spelling alternates are acceptance-only.
* `time.jsonl`       — વાગ્યે / વાગીને / મિનિટ cues, સવારે / બપોરે /
  સાંજે / રાતે time-of-day modifiers, half/quarter time compounds.
* `percent.jsonl`    — ટકા cue acceptance.
* `phone.jsonl`      — Gujarati-digit passthrough (૦-૯,
  U+0AE6..U+0AEF).
* `id.jsonl`         — PAN / Aadhaar / IFSC / policy-number forms.

The reviewer should annotate any entry they want to remove, add, or
correct in this file, then sign off below:

```
Reviewed by:    <name>
Date:           <YYYY-MM-DD>
Sign-off:       <native speaker / dialect — e.g., "Surat Gujarati",
                 "Ahmedabad Gujarati", "Kathiawadi">
Decisions:      <free-form notes — which alternates rejected / added>
```

## Why the block

Gujarati is the second language onboarded under the bn/gu/pa template
stage. The structural shape is direct-copy from Hindi/Marathi —
including Indian grouping, the cardinal scale tree, and the
hundred-compound topology. The only thing that changed is the
lexicon. A lexicon error in this stage propagates to every gold
case downstream, so the cost of catching it pre-merge is much lower
than catching it post-merge.

## How to unblock

After review, delete this file (`git rm REVIEW_REQUIRED.md`) and the
gold-set tests will start running. CI will then enforce the per-class
acceptance bars defined in `tests/grammars/gu/test_*.py` (≥ 98 % per
file, ≥ 200 entries per file, every compound family present — same
contract as the Marathi template).
