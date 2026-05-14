ASR partial frames ──► (emit raw to UI as interim)
                    │
                    └── on endpoint / RNNT-final ──┐
                                                   ▼
                                         finalize_segment(text)
                                                   │
                       ┌───────────────────────────┼───────────────────────────┐
                       ▼                           ▼                           ▼
                detect_script(text)         detect_lang(text, prior)    restore_punct(text)
                       │                           │                           │
                       └────────────┬──────────────┴────────────┬──────────────┘
                                    ▼                           ▼
                            classify_tokens(text, lang)  ──► [(span, class, raw)]
                                    │
                                    ▼
                  for each (span, class, raw):
                      cand = wfst_normalize(raw, class, lang)
                      conf = score(cand, raw, class, context)
                      out = cand if conf >= τ_class else raw
                                    │
                                    ▼
                       splice → normalized_transcript
                                    │
                                    ▼
                   emit { raw_transcript, normalized_transcript,
                          spans: [{class, raw, norm, conf}] }