"""Deterministic hint-versus-question literal conflicts (the ``shared.hint_conflict`` frame).

A BIRD hint sometimes writes a literal that contradicts the question it annotates: a year
inside a point literal that differs from the question's only year, or a comparator value a
power of ten away from the question's quantifier. On BIRD train (dev databases excluded) the
reference follows the question in both classes (year about 14:2, magnitude 4:0). Comparator
inclusivity ("two or more" against ``> 2``) is deliberately not a class: there the reference
follows the hint about 3:1.

The detector reads only the question text and its hint. It never sees a question id, the
data or a reference answer; the frame it feeds asks the model to settle the literal with the
data. A question without a conflict yields nothing, so its messages stay byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .question import HINT_SEPARATOR

__all__ = ["IMPLEMENTATION_VERSION", "Conflict", "detect"]

IMPLEMENTATION_VERSION = "hint-conflicts-2"

_YEAR = r"(?:1[89]\d\d|20\d\d)"
_QUESTION_YEAR = re.compile(r"(?<!\d)(" + _YEAR + r")(?!\d)")
# BIRD hints write comparators with inner spaces ("HCT > = 52"); fold them first so a range
# bound is never mistaken for a point equality.
_SPACED_COMPARATOR = re.compile(r"([<>!=])\s+=")
# A year inside a point literal: after '=' or LIKE, optionally quoted, as the literal's head.
_POINT_YEAR = re.compile(
    r"((?:[\w.]+|`[^`]+`|\[[^\]]+\])\s*)?(?<![<>!=])(==?|\bLIKE\b)\s*(['\"]?)%?(" + _YEAR + r")([^'\"\s;,)]*)", re.I)
_SEASON_TAIL = re.compile(r"\s*/|\s*-\s*" + _YEAR + r"(?!\d)")
_UPPER = ("less than", "under", "below", "fewer than", "lower than", "at most", "no more than", "not more than")
_LOWER = ("more than", "over", "greater than", "exceeds", "exceed", "exceeding", "above", "higher than", "at least",
          "no less than", "not less than")
_QUANTIFIER = re.compile(r"\b(" + "|".join(sorted(_UPPER + _LOWER, key=len, reverse=True)) + r")\s+\$?(\d[\d,]*(?:\.\d+)?)\b", re.I)
# A comparator value, never the '>' of '<>', a percentage ('10%') or the head of a date ('2000-01-01', '1/2/2000').
_HINT_COMPARATOR = re.compile(r"(?<![<>!=])(>=|<=|>(?!=)|<(?![>=]))\s*['\"]?(\d+(?:\.\d+)?)(?![\d.])(?!\s*%)(?!\s*[-/:]\s*\d)")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Conflict:
    """One literal on which the hint and the question disagree.

    Attributes:
        kind (str): ``year`` or ``number``.
        hint_literal (str): The hint's fragment as written, e.g. ``date = '2012-02-22'`` or ``< 10``.
        question_literal (str): The question's fragment, e.g. ``2011`` or ``under 100``.
    """

    kind: str
    hint_literal: str
    question_literal: str


def _split(question: str) -> tuple[str, str]:
    raw, _separator, hint = str(question or "").partition(HINT_SEPARATOR)
    return raw, hint


def _numbers(text: str) -> set[float]:
    values = set()
    for match in _NUMBER.finditer(text):
        try:
            values.add(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def _clause(hint: str, at: int) -> str:
    """The ``;``-separated hint clause around position ``at``."""
    start = hint.rfind(";", 0, at) + 1
    end = hint.find(";", at)
    return hint[start:end if end >= 0 else len(hint)]


def _year_conflicts(raw: str, hint: str) -> list[Conflict]:
    years = set(_QUESTION_YEAR.findall(raw))
    if len(years) != 1:
        return []
    (year,) = years
    points: list[tuple[str, str]] = []
    for match in _POINT_YEAR.finditer(hint):
        if _SEASON_TAIL.match(match.group(5)):
            continue    # a season label such as '2015/2016' maps a year; it is not a typo
        end, quote = match.end(), match.group(3)
        if quote:       # a quoted literal is shown whole, up to its closing quote ('1928 Summer')
            closing = hint.find(quote, match.end(4))
            end = closing + 1 if closing >= 0 else len(hint)
        points.append((match.group(4), hint[match.start():end].strip()))
    if not points or any(point == year for point, _text in points):
        return []
    return [Conflict("year", points[0][1], year)]


def _magnitude_conflicts(raw: str, hint: str) -> list[Conflict]:
    asked = _numbers(raw)
    found: list[Conflict] = []
    for quantifier in _QUANTIFIER.finditer(raw):
        try:
            wanted = float(quantifier.group(2).replace(",", ""))
        except ValueError:
            continue
        if wanted < 1 or wanted != int(wanted):
            continue
        upper = quantifier.group(1).lower() in _UPPER
        for comparator in _HINT_COMPARATOR.finditer(hint):
            value = float(comparator.group(2))
            if value < 1 or value != int(value) or value in asked:
                continue
            if comparator.group(1).startswith("<") != upper:
                continue    # the comparator bounds the other side; it restates no quantifier of this question
            if wanted not in _numbers(_clause(hint, comparator.start())):
                continue    # the comparator's own hint clause must restate the question's number in its prose
            if value in (wanted * 10, wanted * 100, wanted / 10, wanted / 100):
                literal = f"{comparator.group(1)} {comparator.group(2)}"
                conflict = Conflict("number", literal, quantifier.group(0))
                if conflict not in found:
                    found.append(conflict)
    return found


def detect(question: str) -> list[Conflict]:
    """Find the hint literals that contradict the question's own literal.

    Args:
        question (str): The canonical question string, hint appended after ``question.HINT_SEPARATOR``.

    Returns:
        list[Conflict]: The year conflict (at most one) then the magnitude conflicts; empty when the question has
            no hint or nothing fires.
    """
    raw, hint = _split(question)
    if not hint.strip():
        return []
    hint = _SPACED_COMPARATOR.sub(r"\1=", hint)
    return _year_conflicts(raw, hint) + _magnitude_conflicts(raw, hint)
