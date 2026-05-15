# itn_service

Deterministic Indic inverse text normalization (ITN) for live telephony ASR
(IndicConformer 600M). WFST-first, Pynini/OpenFst-backed, ICU/CLDR for
locale rendering. No LLM in the live path.

See [docs/implementation_bluprint_INR.md](../docs/implementation_bluprint_INR.md)
for the full design and `CONTRIBUTING.md` for the invariants this service
maintains. For the current gap between the intended design and the live
request path, see
[docs/itn_live_path_gap_analysis.md](../docs/itn_live_path_gap_analysis.md).

## Layout

```
itn_service/
  configs/        # policy.yaml, locales.yaml, thresholds.yaml (single source of truth)
  grammars/       # common/ + per-language Pynini grammars (hi, mr, bn, ta, te, kn, ml, gu, pa, ur)
  runtime/        # streaming-safe normalizer, contract, gating, FAR cache
  cxx_runtime/    # C++ serving path (Sparrowhawk-style)
  tests/          # gold/<lang>, regression/, latency/
  tools/          # export_far.sh, benchmark_latency.py, build_gold_from_csv.py
```

## Install (dev)

```bash
pip install -e .
pytest
```

`pip install -e .[dev]` adds NeMo, torch, ruff, and mypy for grammar
authoring and lint/type checks.
