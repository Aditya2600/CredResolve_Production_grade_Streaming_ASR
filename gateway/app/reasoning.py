from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ReasoningResult:
    text: str
    source: str


class CallReasoner:
    """Gemini-backed reasoning helper for debt-collection call turns."""

    def __init__(self, api_key: str, model: str, timeout_ms: int) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.timeout = max(0.3, timeout_ms / 1000.0)

    async def respond(
        self,
        *,
        borrower_text: str,
        callee_name: str,
        amount_due: str,
        history: list[dict[str, str]],
    ) -> ReasoningResult:
        borrower_text = (borrower_text or "").strip()
        if not borrower_text:
            return ReasoningResult(
                text="माफ़ कीजिए, आपकी आवाज़ साफ़ नहीं आई। क्या आप फिर से दोहरा सकते हैं?",
                source="fallback",
            )

        if not self.api_key:
            return ReasoningResult(
                text=self._fallback_reply(borrower_text, callee_name, amount_due),
                source="fallback",
            )

        prompt = self._build_prompt(
            borrower_text=borrower_text,
            callee_name=callee_name,
            amount_due=amount_due,
            history=history,
        )
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "maxOutputTokens": 220,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                text = self._extract_text(data).strip()
                if text:
                    return ReasoningResult(text=text, source="gemini")
        except Exception:
            pass

        return ReasoningResult(
            text=self._fallback_reply(borrower_text, callee_name, amount_due),
            source="fallback",
        )

    def _build_prompt(
        self,
        *,
        borrower_text: str,
        callee_name: str,
        amount_due: str,
        history: list[dict[str, str]],
    ) -> str:
        lines = [
            "You are Radha, a debt collection call agent for SMFG India Credit Company.",
            "Return ONLY one conversational line in Hindi (Devanagari), no labels, no markdown.",
            "Tone: polite, firm, short, realistic phone-call style.",
            "Rules:",
            "1) Continue exactly from borrower's latest utterance.",
            "2) If borrower confirms identity, move to EMI reminder with amount and due date 5 फरवरी 2026.",
            "3) If borrower agrees to keep balance, thank and close politely.",
            "4) If not clear, ask them to repeat.",
            "5) Keep under 35 words.",
            f"Callee name: {callee_name}",
            f"EMI amount: ₹{amount_due}",
            "Reference style examples include lines like:",
            "- नमस्ते। मैं SMFG से राधा बोल रही हूँ। क्या मैं ... से बात कर रहा हूँ?",
            "- ... लोन ईएमआई की नियत तारीख नजदीक आ रही है...",
            "- धन्यवाद..., आपका दिन शुभ हो।",
            "Conversation so far:",
        ]

        for item in history[-8:]:
            role = "Agent" if item.get("role") == "agent" else "Borrower"
            text = (item.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        lines.append(f"Borrower latest: {borrower_text}")
        lines.append("Agent next line:")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        snippets = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
        return " ".join(snippets).strip()

    @staticmethod
    def _fallback_reply(borrower_text: str, callee_name: str, amount_due: str) -> str:
        text = borrower_text.lower()
        yes_tokens = ("हाँ", "han", "ha", "yes", " बोल", "speaking", "main")
        no_tokens = ("नहीं", "गलत", "not", "wrong")

        if any(token in text for token in no_tokens):
            return "क्या आप कृपया बता सकते हैं कि उनसे बात करने का सही समय क्या रहेगा?"

        if any(token in text for token in yes_tokens):
            return (
                f"{callee_name} जी, SMFG के साथ आपके ₹{amount_due} के लोन ईएमआई की नियत तारीख 5 फरवरी 2026 है। "
                "क्या आप अपने खाते में पर्याप्त शेष राशि रख पाएंगे?"
            )

        if "भर" in text or "pay" in text or "रख" in text:
            return (
                f"धन्यवाद {callee_name} जी, समय पर भुगतान से आपका पुनर्भुगतान रिकॉर्ड मजबूत रहेगा। "
                "SMFG से बात करने के लिए धन्यवाद।"
            )

        return "माफ़ कीजिए, आपकी आवाज़ साफ़ नहीं आई। क्या आप फिर से दोहरा सकते हैं?"
