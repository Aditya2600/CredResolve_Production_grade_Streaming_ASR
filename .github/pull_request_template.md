<!--
PR template. Sections may be deleted when not applicable, but the
"Language reviewer sign-off" block must stay on any PR that touches
`itn_service/grammars/<lang>/` or `itn_service/tests/gold/<lang>/`.
-->

## Summary

<!-- 1–3 bullets on what changes and why. -->

## Test plan

- [ ] `pytest itn_service/tests` (or the targeted subset for the PR scope)
- [ ] If shared runtime changed: `pytest itn_service/tests/regression/` —
      both `test_hi_gold_no_regression.py` and
      `test_all_langs_gold_no_regression.py` must stay green
- [ ]

## Language reviewer sign-off

Required on any PR that adds, edits, or deletes content under
`itn_service/grammars/<lang>/` or `itn_service/tests/gold/<lang>/`.
Tick the row(s) for the language(s) the PR touches; leave the rest.

A new language gold corpus is gated by its
`tests/gold/<lang>/REVIEW_REQUIRED.md` marker — CI will fail until the
named reviewer signs off **in the marker file** and removes it. The
checkbox here records the human approval; the marker file records the
detailed sign-off (dialect, decisions, rejected alternates).

| Lang | Native-speaker reviewer (handle) | Sign-off in marker? | This PR ok? |
|------|----------------------------------|---------------------|-------------|
| `hi` |                                  | n/a (marker absent) | [ ]         |
| `mr` |                                  | [ ]                 | [ ]         |
| `bn` |                                  | [ ]                 | [ ]         |
| `gu` |                                  | [ ]                 | [ ]         |
| `pa` |                                  | [ ]                 | [ ]         |

If the PR adds a new language not listed above, add a row, create a
`tests/gold/<lang>/REVIEW_REQUIRED.md` marker, and a
`tests/grammars/<lang>/test_review_block.py` guard following the
existing per-language pattern (see `mr` / `bn` / `gu` / `pa`).

## Notes for reviewers

<!-- Anything reviewers need to know that isn't obvious from the diff:
risky migrations, follow-up PRs, on-call considerations, etc. -->
