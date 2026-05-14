# Marathi gold sets — native-speaker review block

**PR merge is blocked until this file is removed.** The corresponding
test, `tests/grammars/mr/test_review_block.py::test_no_marathi_review_required_marker`,
fails while this file exists, so any CI configured to gate on tests
will refuse to merge.

## What this gates

The Marathi gold sets in this directory were drafted by an automated
template-port from the Hindi gold sets (stage 2-4 template). Marathi
number words, half / quarter compounds, month names, time cues,
currency cues, and percent cues are lexically distinct from Hindi
even though both languages share the Devanagari script, so the gold
sets must be reviewed by a native Marathi speaker before they can
serve as the truth source for grammar acceptance.

## Reviewer checklist

For each file in this directory, the reviewer must verify:

* `cardinal.jsonl`   — every 0..99 entry, every hundred compound
  (`एकशे` ... `नऊशे`), every scale combination
  (हजार/लाख/कोटी/अब्ज), every half/quarter compound family
  (सव्वा / दीड / अडीच / साडे / पावणे), and every dialectal spelling
  alternate listed in the grammar.
* `currency.jsonl`   — rupee / paise lexicon, post-cue / pre-cue
  word order, Indian-grouped output (1,25,000).
* `date.jsonl`       — Marathi month names (especially जुलै,
  ऑगस्ट, सप्टेंबर, ऑक्टोबर, नोव्हेंबर, डिसेंबर), spelling alternates
  (फेब्रवारी vs फेब्रुवारी, etc.).
* `time.jsonl`       — `वाजता` / `वाजून` / `मिनिटे` cues,
  सकाळी / दुपारी / संध्याकाळी / रात्री time-of-day mappings,
  half/quarter time compounds.
* `percent.jsonl`    — `टक्के` / `टक्का` cue acceptance.
* `phone.jsonl`      — Devanagari-digit passthrough.
* `id.jsonl`         — PAN / Aadhaar / IFSC / policy-number forms.

The reviewer should annotate any entry they want to remove, add, or
correct in this file, then sign off below:

```
Reviewed by:    <name>
Date:           <YYYY-MM-DD>
Sign-off:       <native speaker / dialect — e.g., "Pune dialect",
                 "Vidarbha dialect">
Decisions:      <free-form notes — which alternates rejected / added>
```

## Why the block

Stages 2-4 in the original implementation blueprint established a
per-language template (lexicon + grammar topology); the Marathi work
was scoped to validate that the template is reusable. The grammar
shape is direct-copy from Hindi by design — the only thing that
changed is the lexicon. A lexicon error in this stage propagates to
every gold case downstream, so the cost of catching it pre-merge is
much lower than catching it post-merge.

## How to unblock

After review, delete this file (`git rm REVIEW_REQUIRED.md`) and
update the test gate. CI will then run the gold-set tests against
the reviewed corpus.
