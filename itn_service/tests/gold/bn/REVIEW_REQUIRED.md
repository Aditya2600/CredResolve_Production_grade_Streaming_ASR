# Bengali gold sets — native-speaker review block

**PR merge is blocked until this file is removed.** The corresponding
test, `tests/grammars/bn/test_review_block.py::test_no_bengali_review_required_marker`,
fails while this file exists, so any CI configured to gate on tests
will refuse to merge.

## What this gates

The Bengali gold sets in this directory were drafted by an automated
template-port from the Hindi / Marathi gold sets. Bengali number
words, scale words (লক্ষ / কোটি), half / quarter compounds, month
names, time cues, percent cues, and — critically — the dual currency
cue (টাকা vs রুপি) are lexically distinct from Hindi/Marathi, so the
gold sets must be reviewed by a native Bengali speaker before they
can serve as the truth source for grammar acceptance.

## Special handling for this language

* **Currency cue policy.** Per `configs/locales.yaml`, the bn-IN
  default accepts EITHER টাকা or রুপি as the currency cue, with ₹ as
  the canonical output. The bn-BD variant accepts টাকা / ৳ / Tk /
  BDT and emits ৳. Reviewers must confirm that the cue lists match
  the dialect / region they are sign-off authority for; in
  particular, flag any টাকা that should be interpreted as BDT in an
  IN-tenant call (this is a known ambiguity for cross-border
  remittance scenarios and should land as a tenant-overridable
  policy, not a grammar branch).
* **Native digit policy.** Bengali numerals (০-৯, U+09E6..U+09EF)
  must be accepted as input and canonicalised to Latin digits
  (per `configs/locales.yaml § numbering_systems`). Reviewers should
  confirm at least one gold case per file exercises a Bengali-digit
  input.

## Reviewer checklist

For each file in this directory, the reviewer must verify:

* `cardinal.jsonl`   — every 0..99 entry, every hundred compound
  (এক শ … নয় শ / একশো … নশো — note both shapes occur in spoken
  Bengali; the gold set must pick one canonical for the WFST input
  vocabulary and document the other as a recognised alternate),
  every scale combination (হাজার / লক্ষ / কোটি), half/quarter
  compound family (দেড় / আড়াই / সোয়া / সাড়ে / পৌনে), and every
  dialectal spelling alternate listed in the grammar.
* `currency.jsonl`   — টাকা vs রুপি acceptance per the bn-IN
  policy; ৳ canonical output for bn-BD cases; পয়সা / পয়সে paise
  cue; Indian-grouped output (1,25,000) for INR cases, plain
  thousands grouping (1,250,000) for BDT cases.
* `date.jsonl`       — Bengali month names (especially জানুয়ারি,
  ফেব্রুয়ারি, এপ্রিল, জুলাই, আগস্ট, সেপ্টেম্বর, অক্টোবর, নভেম্বর,
  ডিসেম্বর). Spelling alternates (জানুয়ারী / জানুয়ারি, etc.) are
  acceptance-only, not generation.
* `time.jsonl`       — টা / টার সময় / মিনিট cues; সকাল / দুপুর /
  বিকেল / সন্ধ্যা / রাত time-of-day modifiers; half/quarter time
  compounds.
* `percent.jsonl`    — শতাংশ / পার্সেন্ট cue acceptance.
* `phone.jsonl`      — Bengali-digit passthrough (০-৯).
* `id.jsonl`         — PAN / Aadhaar / IFSC / policy-number forms.

The reviewer should annotate any entry they want to remove, add, or
correct in this file, then sign off below:

```
Reviewed by:    <name>
Date:           <YYYY-MM-DD>
Region:         <IN | BD>
Sign-off:       <native speaker / dialect — e.g., "Kolkata Bangla",
                 "Dhaka Bangla">
Decisions:      <free-form notes — which alternates rejected / added>
```

## Why the block

Bengali is the first language onboarded under the bn/gu/pa template
stage. Two failure modes are easy to hit and very expensive to undo
after the gold sets are committed as truth:

1. Choosing the wrong canonical for a hundred-compound family
   (একশো vs এক শ) — every cardinal gold case inherits the choice.
2. Mis-routing টাকা between INR-canonical and BDT-canonical paths.
   The `bn-IN` / `bn-BD` split is the policy knob; the gold sets
   must exercise both sides.

## How to unblock

After review, delete this file (`git rm REVIEW_REQUIRED.md`) and the
gold-set tests will start running. CI will then enforce the per-class
acceptance bars defined in `tests/grammars/bn/test_*.py` (≥ 98 % per
file, ≥ 200 entries per file, every compound family present — same
contract as the Marathi template).
