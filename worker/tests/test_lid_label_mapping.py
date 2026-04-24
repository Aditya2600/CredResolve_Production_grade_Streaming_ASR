from __future__ import annotations

import pytest

from worker.app.lid import BaseLanguageDetector


@pytest.mark.parametrize(
    ("raw_label", "supported_languages", "expected"),
    [
        ("telugu", {"te"}, "te"),
        ("bengali", {"bn"}, "bn"),
        ("gujarati", {"gu"}, "gu"),
        ("kannada", {"kn"}, "kn"),
        ("malayalam", {"ml"}, "ml"),
        ("odia", {"or"}, "or"),
        ("oriya", {"or"}, "or"),
        ("punjabi", {"pa"}, "pa"),
        ("assamese", {"as"}, "as"),
        ("urdu", {"ur"}, "ur"),
        ("unknown-language", {"hi", "te"}, None),
    ],
)
def test_map_to_supported_code_handles_indic_language_aliases(
    raw_label: str,
    supported_languages: set[str],
    expected: str | None,
) -> None:
    assert BaseLanguageDetector.map_to_supported_code(raw_label, supported_languages) == expected
