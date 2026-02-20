from __future__ import annotations

import re


WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lightweight text normalization safe for multilingual output."""
    out = text.replace("\u200c", " ").replace("\u200d", " ")
    out = out.replace("|", " ").replace("।", " । ")
    out = WS_RE.sub(" ", out).strip()
    return out


def normalize_tokens(tokens: list[str]) -> list[str]:
    return [normalize_text(token) for token in tokens if normalize_text(token)]
