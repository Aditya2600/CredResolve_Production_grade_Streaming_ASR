from __future__ import annotations

from dataclasses import dataclass

from asr.models.script_head import ScriptHead
from asr.models.tokenlid_head import TokenLIDHead
from asr.postprocess.normalize import normalize_text
from asr.postprocess.transliteration import detect_script, transliterate_telugu_to_devanagari


@dataclass(frozen=True)
class ScriptFixEvent:
    token_before: str
    token_after: str
    token_lang: str
    token_script: str
    action: str


def fix_script_confusion(
    tokens: list[str],
    token_lang: list[str],
    token_script: list[str],
) -> tuple[list[str], list[ScriptFixEvent]]:
    if not (len(tokens) == len(token_lang) == len(token_script)):
        raise ValueError("tokens/token_lang/token_script lengths must match")

    fixed: list[str] = []
    events: list[ScriptFixEvent] = []

    for token, lang, script in zip(tokens, token_lang, token_script):
        new_token = token
        action = "keep"

        # Fix only the target bug: Hindi token emitted in Telugu script.
        if lang == "hi" and script == "telugu":
            new_token = transliterate_telugu_to_devanagari(token)
            action = "telugu_to_devanagari"

        fixed.append(new_token)
        events.append(
            ScriptFixEvent(
                token_before=token,
                token_after=new_token,
                token_lang=lang,
                token_script=script,
                action=action,
            )
        )

    return fixed, events


def fix_text(
    text: str,
    token_lang: list[str] | None = None,
    token_script: list[str] | None = None,
    token_lid_head: TokenLIDHead | None = None,
    script_head: ScriptHead | None = None,
) -> tuple[str, list[ScriptFixEvent]]:
    tokens = [t for t in normalize_text(text).split(" ") if t]
    if not tokens:
        return "", []

    lid_head = token_lid_head or TokenLIDHead()
    scr_head = script_head or ScriptHead()

    if token_lang is None:
        token_lang = [pred.language for pred in lid_head.predict_tokens(tokens)]
    if token_script is None:
        token_script = [pred.script for pred in scr_head.predict_tokens(tokens)]
    else:
        token_script = [s or detect_script(tok) for tok, s in zip(tokens, token_script)]

    fixed_tokens, events = fix_script_confusion(tokens, token_lang, token_script)
    return normalize_text(" ".join(fixed_tokens)), events
