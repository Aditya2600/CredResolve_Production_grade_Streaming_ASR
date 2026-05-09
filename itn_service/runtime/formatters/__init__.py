"""Deterministic post-WFST formatters for high-stakes spans.

These formatters run *after* the WFST classifier and apply pure regex +
structural validators (with checksums where mandated). They never
fuzzy-match, never guess missing digits, and never silently rewrite an
identifier. The default policy for any uncertainty is to return ``None``
so the caller emits the raw text instead.

Modules:
    * :mod:`.phone_in`              — Indian mobile (+91 XXXXX XXXXX).
    * :mod:`.id_pan_aadhaar_ifsc`   — PAN, Aadhaar (Verhoeff), IFSC,
                                       cue-gated generic alphanumeric IDs.
"""
