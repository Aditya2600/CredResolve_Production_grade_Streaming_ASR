from __future__ import annotations

from pathlib import Path

from worker.app.context_assembler import build_request_scoped_phrase_pack, parse_biasing_context


def test_parse_biasing_context_normalizes_strings_and_lists():
    context = parse_biasing_context(
        {
            "debtor_name": "  Ravi   Kumar ",
            "agent_name": None,
            "account_terms": [" EMI ", "EMI", " payment link "],
            "amounts": "12500, Rs 8,450 , 12500",
        }
    )

    assert context.debtor_name == "Ravi Kumar"
    assert context.agent_name == ""
    assert context.account_terms == ("EMI", "payment link")
    assert context.amounts == ("12500", "Rs 8,450")
    assert context.provided_fields == ("debtor_name", "account_terms", "amounts")


def test_build_request_scoped_phrase_pack_merges_static_base_with_ranked_dynamic_terms(tmp_path: Path):
    base_phrase_file = tmp_path / "hi.txt"
    base_phrase_file.write_text("loan id_loan id_लोन आईडी\n", encoding="utf-8")

    context = parse_biasing_context(
        {
            "debtor_name": "Ravi Kumar",
            "lender": "SMFG India Credit",
            "product": "personal loan",
            "city": "Jaipur",
            "branch": "MI Road",
            "campaign_vocabulary": ["callback", "loan"],
        }
    )

    pack = build_request_scoped_phrase_pack(
        context=context,
        base_phrase_file=base_phrase_file,
        max_dynamic_phrases=3,
    )

    assert pack is not None
    assert pack.dynamic_context_present is True
    assert pack.dynamic_context_used is True
    assert pack.phrase_count_before_pruning >= 6
    assert pack.phrase_count_after_pruning == 3
    assert pack.top_phrases[0] == "Ravi Kumar"
    assert any(line.startswith("loan id_") for line in pack.lines)
    assert any(line.startswith("Ravi Kumar_") for line in pack.lines)


def test_build_request_scoped_phrase_pack_generates_conservative_amount_and_date_variants():
    context = parse_biasing_context(
        {
            "amounts": ["Rs 12,500"],
            "dates": ["21/04/2026"],
        }
    )

    pack = build_request_scoped_phrase_pack(
        context=context,
        base_phrase_file=None,
        max_dynamic_phrases=8,
    )

    assert pack is not None
    rendered = "\n".join(pack.lines)
    assert "Rs 12,500" in rendered
    assert "12,500" in rendered
    assert "12500" in rendered
    assert "21/04/2026" in rendered
    assert "2026-04-21" in rendered


def test_build_request_scoped_phrase_pack_adds_hindi_script_variants_for_latin_names():
    context = parse_biasing_context(
        {
            "debtor_name": "Aditya",
            "agent_name": "Ravi Kumar",
        }
    )

    pack = build_request_scoped_phrase_pack(
        context=context,
        base_phrase_file=None,
        max_dynamic_phrases=8,
        language="hi",
    )

    assert pack is not None
    rendered = "\n".join(pack.lines)
    assert "Aditya" in rendered
    assert "आदित्य" in rendered
    assert "Ravi Kumar" in rendered
    assert "रवि कुमार" in rendered


def test_build_request_scoped_phrase_pack_keeps_non_hindi_requests_unchanged():
    context = parse_biasing_context({"debtor_name": "Aditya"})

    pack = build_request_scoped_phrase_pack(
        context=context,
        base_phrase_file=None,
        max_dynamic_phrases=8,
        language="en",
    )

    assert pack is not None
    rendered = "\n".join(pack.lines)
    assert "Aditya" in rendered
    assert "आदित्य" not in rendered


def test_build_request_scoped_phrase_pack_adds_hindi_amount_and_date_variants():
    context = parse_biasing_context(
        {
            "amounts": ["Rs 12,500"],
            "dates": ["21/04/2026"],
        }
    )

    pack = build_request_scoped_phrase_pack(
        context=context,
        base_phrase_file=None,
        max_dynamic_phrases=8,
        language="hi",
    )

    assert pack is not None
    rendered = "\n".join(pack.lines)
    assert "12,500" in rendered
    assert "१२,५००" in rendered
    assert "१२,५०० रुपये" in rendered
    assert "२१/०४/२०२६" in rendered
    assert "21 अप्रैल 2026" in rendered
    assert "२१ अप्रैल २०२६" in rendered
