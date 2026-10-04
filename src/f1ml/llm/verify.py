"""Check an LLM narrative against the facts it was given.

Two rules:
  1. Every number in the text must appear in the facts (incl. inside strings
     such as '1:27.452' or '64%'). Number words (two, three, ...) count too.
  2. Every known driver surname / team name in the text must appear in the facts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_NUM = re.compile(r"(?<![\d.])(\d+(?:[.,]\d+)*)")   # also catches P3, Q2, F1
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
# Words that are numbers but harmless in any recap ("one of the", "first time")
_SAFE_WORDS = {"one", "first", "second"}
# Always allowed: the 1 in "F1" / "Formula 1"
_ALWAYS_OK = {"1"}


def _norm(n: str) -> str:
    """'1,042' -> '1042'; '27.450' -> '27.45'; '3.0' -> '3'."""
    n = n.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def numbers_in(text: str) -> set[str]:
    found = {_norm(m) for m in _NUM.findall(_ORDINAL.sub(r"\1", text))}
    # '1:27.452' also contributes '1:27.452' parts: 1 and 27.452 (already by regex)
    return found


def number_words_in(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {str(_WORDS[w]) for w in words if w in _WORDS and w not in _SAFE_WORDS}


def _flatten(obj) -> list[str]:
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _flatten(v)] + [str(k) for k in obj]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _flatten(v)]
    return [str(obj)]


@dataclass
class Verdict:
    ok: bool
    unknown_numbers: list[str] = field(default_factory=list)
    unknown_names: list[str] = field(default_factory=list)

    def reason(self) -> str:
        parts = []
        if self.unknown_numbers:
            parts.append(f"numbers not in facts: {self.unknown_numbers}")
        if self.unknown_names:
            parts.append(f"names not in facts: {self.unknown_names}")
        return "; ".join(parts) or "ok"


def verify(text: str, facts: dict, roster: set[str] | None = None) -> Verdict:
    fact_strings = _flatten(facts)
    fact_blob = " ".join(fact_strings)
    allowed_numbers = numbers_in(fact_blob)
    for s in fact_strings:                      # 64.0 in facts also allows "64"
        try:
            allowed_numbers.add(_norm(str(round(float(s)))))
        except (ValueError, OverflowError):
            pass

    used = numbers_in(text) | number_words_in(text)
    unknown_numbers = sorted(n for n in used if n not in allowed_numbers | _ALWAYS_OK)

    unknown_names = []
    if roster:
        for name in roster:
            if re.search(rf"\b{re.escape(name)}\b", text) and name not in fact_blob:
                unknown_names.append(name)

    return Verdict(ok=not unknown_numbers and not unknown_names,
                   unknown_numbers=unknown_numbers, unknown_names=sorted(unknown_names))


def fact_hash(facts: dict, prompt_version: str, model: str) -> str:
    import hashlib

    blob = json.dumps({"facts": facts, "prompt": prompt_version, "model": model},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
