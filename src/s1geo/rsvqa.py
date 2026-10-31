"""RSVQAxBEN helpers: class vocabulary per level and parsing of land-cover (LC) answer strings.

LC answers list classes joined by ", " and " and "; some CORINE names contain commas themselves, so answers
are parsed by greedy longest-match against the vocabulary instead of splitting on separators.
"""
from __future__ import annotations

import re

from s1geo.clc import LEVEL_CODES, NAMES

# Spellings used by RSVQAxBEN where they differ from the official CORINE names (lower-cased).
EXTRA_NAMES = ["forests", "maritime wetlands", "artificial areas", "artificial non-agricultural vegetated areas",
               "forest and semi natural areas", "forest and semi-natural areas"]


def candidate_names() -> list[str]:
    names = {NAMES[c].lower() for lvl in (1, 2, 3) for c in LEVEL_CODES[lvl]} | set(EXTRA_NAMES)
    names |= {n.replace(",", "") for n in names}
    return sorted(names, key=len, reverse=True)


_SEP = re.compile(r"^(,\s*|\s+and\s+|\s+)")


def parse_classes(answer: str, vocab: list[str] | None = None) -> tuple[list[str], str]:
    """Split an LC answer into class names. Returns (classes, leftover text that matched nothing)."""
    text = answer.strip().lower()
    if text in ("none", ""):
        return [], ""
    vocab = vocab or candidate_names()
    out, rest = [], text
    while rest:
        m = _SEP.match(rest)
        if m:
            rest = rest[m.end():]
            continue
        hit = next((v for v in vocab if rest.startswith(v)), None)
        if hit is None:
            return out, rest
        out.append(hit)
        rest = rest[len(hit):]
    return out, ""
