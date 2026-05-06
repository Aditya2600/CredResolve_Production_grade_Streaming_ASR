"""Per-language WFST pipeline.

Wraps a build-time-compiled FAR archive (see ``itn_service.compile``)
behind a tiny request-path API. Construction loads the FAR once and
caches every entry on the instance; ``normalize_span`` performs a
single FST composition per call. Per ``CONTRIBUTING.md`` invariant 3,
no graph construction happens in the request path.

Typical use::

    pipe = WFSTPipeline("hi")
    pipe.normalize_span("एक सौ पच्चीस", "cardinal")  # -> "125"
    pipe.normalize_span("बारह दशमलव पाँच", "decimal") # -> "12.5"
    pipe.normalize_span("नमस्ते", "cardinal")          # -> None (no parse)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pynini


# --- supported semiotic classes ----------------------------------------------

# Each class maps to two FAR entry names: the bare normaliser (raw
# Latin output) and the NeMo-tagged classifier wrapper. Keeping the
# names in lockstep with ``compile.py`` so the contract is verified at
# load time.
_BARE_FST_NAME: Final[dict[str, str]] = {
    "cardinal": "CARDINAL",
    "decimal": "DECIMAL",
}

_CLASSIFIER_FST_NAME: Final[dict[str, str]] = {
    "cardinal": "CARDINAL_CLASSIFIER",
    "decimal": "DECIMAL_CLASSIFIER",
}


def _default_far_root() -> Path:
    """Default location of compiled FARs: ``itn_service/compiled_grammars``.

    Mirrors the path written by ``itn_service.compile`` so the runtime
    finds the artefact without explicit configuration.
    """
    return Path(__file__).resolve().parent.parent / "compiled_grammars"


class WFSTPipeline:
    """One pipeline per language; thread-safety: read-only after init.

    Args:
        lang: 2-letter language code (matches ``compiled_grammars/<lang>.far``).
        far_root: directory containing the FAR archive. Defaults to
            ``itn_service/compiled_grammars``.

    Raises:
        FileNotFoundError: when the FAR for ``lang`` has not been built.
        KeyError: when the FAR is present but missing a required named
            entry — usually a sign that ``compile.py`` and
            ``wfst_pipeline.py`` have drifted apart.
    """

    __slots__ = ("lang", "_far_path", "_bare", "_classifier")

    def __init__(self, lang: str, far_root: Path | None = None) -> None:
        root = far_root if far_root is not None else _default_far_root()
        far_path = root / f"{lang}.far"
        if not far_path.exists():
            raise FileNotFoundError(
                f"FAR for {lang!r} not found at {far_path}; "
                f"run `python -m itn_service.compile --lang {lang}` first"
            )
        self.lang: str = lang
        self._far_path: Path = far_path

        # Slurp every named entry into a dict — opening / iterating a
        # FAR is a one-shot operation in the OpenFst Python bindings.
        loaded: dict[str, pynini.Fst] = {}
        far = pynini.Far(str(far_path), mode="r")
        try:
            for name, fst in far:
                loaded[name] = fst
        finally:
            far.close()

        # Bind by class so the request path doesn't pay a dict lookup
        # plus a string compare per call.
        self._bare: dict[str, pynini.Fst] = {}
        self._classifier: dict[str, pynini.Fst] = {}
        for cls, name in _BARE_FST_NAME.items():
            try:
                self._bare[cls] = loaded[name]
            except KeyError as e:
                raise KeyError(
                    f"FAR {far_path} is missing entry {name!r} required "
                    f"for class {cls!r}"
                ) from e
        for cls, name in _CLASSIFIER_FST_NAME.items():
            if name in loaded:
                self._classifier[cls] = loaded[name]

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @property
    def supported_classes(self) -> tuple[str, ...]:
        """Classes for which a ``normalize_span`` call will be answered."""
        return tuple(self._bare)

    def normalize_span(self, raw: str, cls: str) -> str | None:
        """Normalise ``raw`` against the ``cls`` grammar.

        Returns:
            The canonical Latin-digit form on success, or ``None`` when
            no FST path matches the input. Returning ``None`` is the
            signal upstream uses to defer (per the plan's confidence-
            gating policy); the caller decides whether to fall through
            to a different class or surface the raw text unchanged.

        Raises:
            ValueError: ``cls`` is not in :attr:`supported_classes`.
        """
        try:
            fst = self._bare[cls]
        except KeyError as e:
            raise ValueError(
                f"unknown class {cls!r}; supported: {self.supported_classes}"
            ) from e
        return _try_compose(raw, fst)

    def classify_span(self, raw: str, cls: str) -> str | None:
        """Like :meth:`normalize_span` but returns the NeMo-tagged form
        ``cls { value: "..." }`` directly, for callers that want to
        feed the classifier output into a downstream verbaliser /
        permutation generator.

        Returns ``None`` when ``cls`` has no classifier entry in the FAR
        or when no FST path matches.
        """
        try:
            fst = self._classifier[cls]
        except KeyError:
            return None
        return _try_compose(raw, fst)


# ---------------------------------------------------------------------------
# Composition helper.
# ---------------------------------------------------------------------------


def _try_compose(raw: str, fst: pynini.Fst) -> str | None:
    """Compose ``accep(raw) @ fst``; return the best output path or ``None``.

    Handles three failure modes silently:
      * empty input — ``pynini.accep("")`` is the empty acceptor; the
        composition yields the empty FST; we map that to ``None``.
      * no matching path — the composition is non-empty as a graph but
        has no accepting state.
      * multi-path output — we pick the shortest path (deterministic
        for our grammars, since we wire them to be unambiguous).
    """
    if not raw:
        return None
    try:
        composed = pynini.accep(raw) @ fst
    except pynini.FstOpError:
        return None
    if composed.start() == pynini.NO_STATE_ID:
        return None
    try:
        return pynini.shortestpath(composed).string()
    except pynini.FstOpError:
        return None


__all__ = ["WFSTPipeline"]
