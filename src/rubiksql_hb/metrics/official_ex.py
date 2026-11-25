"""The official BIRD EX comparison — the ONE executable EX definition in this repo.

The official evaluator (mini_dev ``evaluation_ex.py``) reduces to
``set(predicted_res) == set(ground_truth_res)`` over raw ``cursor.fetchall()``
tuples: row order and duplicates are free, in-row column order and count are
strict, and values compare by plain Python equality — no rounding and no type
normalization (``118 == 118.0`` because Python says so, not because a
normalizer said so).

This module exists so no second EX implementation with potential subtle
divergence can survive. Every EX number the repo reports calls
:func:`official_ex_match`: eval_bird's per-question score, the lab's
counterfactual candidate scoring (``lab_score``), the gold consistency probe
(``lab_probe``), and the submission rescore (``scripts/rescore_ex.py``).
:func:`row_tuple` unwraps the execution harness's dict-shaped rows back into
the positional tuples the official comparison is defined over; it touches no
value. :func:`result_signature` uses that same unwrapping to compare execution
records with the official row-set semantics.
"""
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["official_ex_match", "result_signature", "row_tuple"]


def result_signature(result: Mapping[str, Any]) -> frozenset[tuple[Any, ...]]:
    """Sign executed rows with official set semantics and preserved column order.

    Args:
        result (Mapping[str, Any]): Execution record with dictionary, positional
            or scalar ``rows``. Missing or ``None`` rows count as empty.

    Returns:
        frozenset[tuple[Any, ...]]: Raw row tuples, without duplicate rows or
            row ordering.
    """
    return frozenset(row_tuple(r) for r in result.get("rows") or [])


def row_tuple(row: Any) -> tuple[Any, ...]:
    """Return one fetched row as the raw positional tuple the official evaluator compares.

    A dict row (this repo's execution harness) keeps insertion order, which is the
    SELECT column order — exactly the tuple ``cursor.fetchall()`` would have yielded.
    A list/tuple row (a raw fetch, e.g. rescore_ex) passes through; a bare scalar
    becomes a 1-tuple. Values are NOT normalized: the comparison owns the semantics.

    Args:
        row: One result row — dict, tuple, list, or scalar.

    Returns:
        tuple: The row's values, positionally.
    """
    if isinstance(row, dict):
        return tuple(row.values())
    if isinstance(row, (list, tuple)):
        return tuple(row)
    return (row,)


def official_ex_match(pred_rows: Iterable[Any] | None, gold_rows: Iterable[Any] | None) -> float:
    """Score a prediction with the official BIRD EX comparison, verbatim.

    ``set(predicted_res) == set(ground_truth_res)``: row order and duplicates are
    free, in-row column order and count are strict, raw values.

    Args:
        pred_rows: Rows the prediction returned; ``None`` counts as empty.
        gold_rows: Rows the gold returned; ``None`` counts as empty.

    Returns:
        float: 1.0 on a match, 0.0 otherwise.
    """
    pred = {row_tuple(r) for r in (pred_rows or [])}
    gold = {row_tuple(r) for r in (gold_rows or [])}
    return 1.0 if pred == gold else 0.0
