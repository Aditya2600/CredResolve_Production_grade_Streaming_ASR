from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tools.normalize_indic_transcripts import (
    DEFAULT_CONFIG_PATH,
    extract_transcript_text,
    infer_language_code_from_text,
    load_language_configs,
    normalize_text_for_language,
    process_rows,
)


def test_marathi_smfg_example_normalizes_to_spelled_out_asr_targets() -> None:
    configs, alias_map = load_language_configs(DEFAULT_CONFIG_PATH)

    result = normalize_text_for_language(
        "मी SMFG इंडिया क्रेडिट कंपनीतून बोलत आहे",
        "MARATHI",
        configs=configs,
        alias_map=alias_map,
    )

    assert result["original_text"] == "मी SMFG इंडिया क्रेडिट कंपनीतून बोलत आहे"
    assert result["asr_l1_text"] == "मी एस एम एफ जी इंडिया क्रेडिट कंपनीतून बोलत आहे"
    assert result["normalized_l2_text"] == "मी SMFG इंडिया क्रेडिट कंपनीतून बोलत आहे"
    assert json.loads(result["entity_map_json"]) == [
        {
            "id": "smfg",
            "kind": "acronym",
            "original": "SMFG",
            "asr": "एस एम एफ जी",
            "normalized": "SMFG",
            "start": 3,
            "end": 7,
        }
    ]


def test_bengali_native_variant_canonicalizes_back_to_smfg() -> None:
    configs, alias_map = load_language_configs(DEFAULT_CONFIG_PATH)

    result = normalize_text_for_language(
        "আমি এসএমএফজি ইন্ডিয়া ক্রেডিট কোম্পানি থেকে বলছি",
        "bengali",
        configs=configs,
        alias_map=alias_map,
    )

    assert result["asr_l1_text"] == "আমি এস এম এফ জি ইন্ডিয়া ক্রেডিট কোম্পানি থেকে বলছি"
    assert result["normalized_l2_text"] == "আমি SMFG ইন্ডিয়া ক্রেডিট কোম্পানি থেকে বলছি"
    entity_map = json.loads(result["entity_map_json"])
    assert entity_map[0]["id"] == "smfg_india_credit_company"
    assert entity_map[0]["original"] == "এসএমএফজি ইন্ডিয়া ক্রেডিট কোম্পানি"


def test_odia_alias_is_supported_and_normalizes_finance_terms() -> None:
    configs, alias_map = load_language_configs(DEFAULT_CONFIG_PATH)

    result = normalize_text_for_language(
        "ମୁଁ ଆପଣଙ୍କୁ କଲ୍ କରୁଛି କାରଣ ଆପଣ SMFG INDIA CREDIT COMPANY ଠାରୁ ନେଇଥିବା ଋଣର EMI ପୈଠ କରିନାହାନ୍ତି।",
        "ODIA",
        configs=configs,
        alias_map=alias_map,
    )

    assert result["asr_l1_text"] == (
        "ମୁଁ ଆପଣଙ୍କୁ କଲ୍ କରୁଛି କାରଣ ଆପଣ ଏସ ଏମ ଏଫ ଜି ଇଣ୍ଡିଆ କ୍ରେଡିଟ୍ କମ୍ପାନୀ ଠାରୁ ନେଇଥିବା ଋଣର ଇଏମଆଇ ପୈଠ କରିନାହାନ୍ତି।"
    )
    assert result["normalized_l2_text"] == (
        "ମୁଁ ଆପଣଙ୍କୁ କଲ୍ କରୁଛି କାରଣ ଆପଣ SMFG ଇଣ୍ଡିଆ କ୍ରେଡିଟ୍ କମ୍ପାନୀ ଠାରୁ ନେଇଥିବା ଋଣର EMI ପୈଠ କରିନାହାନ୍ତି।"
    )
    entity_map = json.loads(result["entity_map_json"])
    assert [item["id"] for item in entity_map] == ["smfg_india_credit_company", "emi"]


def test_malayalam_alias_is_supported_and_normalizes_finance_terms() -> None:
    configs, alias_map = load_language_configs(DEFAULT_CONFIG_PATH)

    result = normalize_text_for_language(
        "SMFG ഇന്ത്യ ക്രെഡിറ്റ് കമ്പനിയിൽ നിന്നുള്ള നിങ്ങളുടെ EMI ഇപ്പോഴും തീർപ്പാക്കാതെ കിടക്കുകയാണ്.",
        "MALAYALAM",
        configs=configs,
        alias_map=alias_map,
    )

    assert result["asr_l1_text"] == (
        "എസ് എം എഫ് ജി ഇന്ത്യ ക്രെഡിറ്റ് കമ്പനിയിൽ നിന്നുള്ള നിങ്ങളുടെ ഇഎംഐ ഇപ്പോഴും തീർപ്പാക്കാതെ കിടക്കുകയാണ്."
    )
    assert result["normalized_l2_text"] == (
        "SMFG ഇന്ത്യ ക്രെഡിറ്റ് കമ്പനിയിൽ നിന്നുള്ള നിങ്ങളുടെ EMI ഇപ്പോഴും തീർപ്പാക്കാതെ കിടക്കുകയാണ്."
    )
    entity_map = json.loads(result["entity_map_json"])
    assert [item["id"] for item in entity_map] == ["smfg", "emi"]


def test_extract_transcript_text_flattens_interaction_transcript_payload() -> None:
    payload = json.dumps(
        {
            "interaction_transcript": [
                {"role": "agent", "en_text": "నమస్కారం. నేను SMFG ఇండియా క్రెడిట్ కంపెనీ నుండి మాట్లాడుతున్నాను."},
                {"role": "user", "en_text": "అవును, చెప్పండి."},
            ]
        },
        ensure_ascii=False,
    )

    assert extract_transcript_text(payload) == (
        "నమస్కారం. నేను SMFG ఇండియా క్రెడిట్ కంపెనీ నుండి మాట్లాడుతున్నాను. అవును, చెప్పండి."
    )


def test_infer_language_code_from_text_handles_missing_language_rows() -> None:
    assert infer_language_code_from_text("నిర్ధారించినందుకు ధన్యవాదాలు. మీ KYC వివరాలను అప్‌డేట్ చేయాలి.") == "te"
    assert infer_language_code_from_text("નમસ્તે. શું હું પરેશ બી દેસાઈ સાથે વાત કરી રહી છું?") == "gu"
    assert infer_language_code_from_text("नमस्ते। मैं राधा बोल रही हूँ।") == "hi"
    assert infer_language_code_from_text("नमस्कार. मी राधा बोलत आहे.") == "mr"


def test_process_rows_supports_nested_json_text_and_preserves_source_columns() -> None:
    configs, alias_map = load_language_configs(DEFAULT_CONFIG_PATH)
    rows = [
        {
            "call_id": "abc123",
            "language": "TELUGU",
            "native_language_transcript": json.dumps(
                {
                    "interaction_transcript": [
                        {"role": "agent", "en_text": "మీ KYC వివరాలు అప్‌డేట్ చేయాలి"}
                    ]
                },
                ensure_ascii=False,
            ),
        }
    ]

    processed = process_rows(
        rows,
        text_column="native_language_transcript",
        language_column="language",
        fixed_language=None,
        configs=configs,
        alias_map=alias_map,
        columns_only=False,
    )

    assert processed == [
        {
            "original_text": "మీ KYC వివరాలు అప్‌డేట్ చేయాలి",
            "asr_l1_text": "మీ కేవైసీ వివరాలు అప్‌డేట్ చేయాలి",
            "normalized_l2_text": "మీ KYC వివరాలు అప్‌డేట్ చేయాలి",
            "entity_map_json": json.dumps(
                [
                    {
                        "id": "kyc",
                        "kind": "finance_term",
                        "original": "KYC",
                        "asr": "కేవైసీ",
                        "normalized": "KYC",
                        "start": 3,
                        "end": 6,
                    }
                ],
                ensure_ascii=False,
            ),
            "call_id": "abc123",
            "language": "TELUGU",
            "native_language_transcript": rows[0]["native_language_transcript"],
        }
    ]


def test_process_rows_fills_missing_language_from_transcript_text() -> None:
    configs, alias_map = load_language_configs(DEFAULT_CONFIG_PATH)
    rows = [
        {
            "call_id": "missing-lang-1",
            "language": None,
            "native_language_transcript": json.dumps(
                {
                    "interaction_transcript": [
                        {"role": "agent", "en_text": "నమస్కారం. నేను రాధను మాట్లాడుతున్నాను."},
                        {"role": "agent", "en_text": "మీ KYC వివరాలను అప్‌డేట్ చేయాలి."},
                    ]
                },
                ensure_ascii=False,
            ),
        }
    ]

    processed = process_rows(
        rows,
        text_column="native_language_transcript",
        language_column="language",
        fixed_language=None,
        configs=configs,
        alias_map=alias_map,
        columns_only=False,
    )

    assert processed[0]["language"] == "TELUGU"
    assert processed[0]["asr_l1_text"] == "నమస్కారం. నేను రాధను మాట్లాడుతున్నాను. మీ కేవైసీ వివరాలను అప్‌డేట్ చేయాలి."
    assert processed[0]["normalized_l2_text"] == "నమస్కారం. నేను రాధను మాట్లాడుతున్నాను. మీ KYC వివరాలను అప్‌డేట్ చేయాలి."


def test_cli_round_trip_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {
                "language": "GUJARATI",
                "transcript": "નમસ્તે. હું SMFG INDIA CREDIT COMPANY માંથી બોલું છું.",
            }
        ]
    ).to_csv(input_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "tools/normalize_indic_transcripts.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    frame = pd.read_csv(output_path)
    assert list(frame.columns[:4]) == [
        "original_text",
        "asr_l1_text",
        "normalized_l2_text",
        "entity_map_json",
    ]
    assert frame.loc[0, "asr_l1_text"] == "નમસ્તે. હું એસ એમ એફ જી ઇન્ડિયા ક્રેડિટ કંપની માંથી બોલું છું."
    assert frame.loc[0, "normalized_l2_text"] == "નમસ્તે. હું SMFG ઇન્ડિયા ક્રેડિટ કંપની માંથી બોલું છું."


def test_cli_round_trip_jsonl_columns_only(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "language": "KANNADA",
                "transcript": "EMI ಅನ್ನು ಕ್ಲಿಯರ್ ಮಾಡದಿದ್ದರೆ ನಿಮ್ಮ CIBIL ಸ್ಕೋರ್ ಮೇಲೆ ಪರಿಣಾಮ ಬೀರುತ್ತದೆ.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "tools/normalize_indic_transcripts.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--columns-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["original_text"] == "EMI ಅನ್ನು ಕ್ಲಿಯರ್ ಮಾಡದಿದ್ದರೆ ನಿಮ್ಮ CIBIL ಸ್ಕೋರ್ ಮೇಲೆ ಪರಿಣಾಮ ಬೀರುತ್ತದೆ."
    assert rows[0]["asr_l1_text"] == "ಇಎಂಐ ಅನ್ನು ಕ್ಲಿಯರ್ ಮಾಡದಿದ್ದರೆ ನಿಮ್ಮ ಸಿಬಿಲ್ ಸ್ಕೋರ್ ಮೇಲೆ ಪರಿಣಾಮ ಬೀರುತ್ತದೆ."
    assert rows[0]["normalized_l2_text"] == "EMI ಅನ್ನು ಕ್ಲಿಯರ್ ಮಾಡದಿದ್ದರೆ ನಿಮ್ಮ CIBIL ಸ್ಕೋರ್ ಮೇಲೆ ಪರಿಣಾಮ ಬೀರುತ್ತದೆ."
    entity_map = json.loads(rows[0]["entity_map_json"])
    assert entity_map == [
        {
            "id": "emi",
            "kind": "finance_term",
            "original": "EMI",
            "asr": "ಇಎಂಐ",
            "normalized": "EMI",
            "start": 0,
            "end": 3,
        },
        {
            "id": "cibil",
            "kind": "finance_term",
            "original": "CIBIL",
            "asr": "ಸಿಬಿಲ್",
            "normalized": "CIBIL",
            "start": rows[0]["original_text"].index("CIBIL"),
            "end": rows[0]["original_text"].index("CIBIL") + len("CIBIL"),
        },
    ]
