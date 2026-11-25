"""Inline N-pass judge majority (``RUBIKSQL_JUDGE_PASSES``).

One question's candidate pool can be judged by N independent judge pipelines that run
concurrently (`rubiksql_hb.vote`). Each pipeline is the ordinary single-pass path: the
judge, the reviewer, the consensus guard, its answerless fallback and the post-review
hooks. This module owns the pass count and the majority that picks one pipeline's answer.
The rule is the offline majority's: passes are grouped by their final execution result
(official EX set semantics); a pass without an executed answer forms its own group; the
largest group wins and a tie goes to the group whose earliest member is earliest.
"""

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

#: Environment variable naming the number of concurrent judge pipelines per question.
PASSES_ENV = "RUBIKSQL_JUDGE_PASSES"

#: Recorded on every majority so a later rule change is distinguishable in saved rows.
IMPLEMENTATION_VERSION = "judge-majority/1"

#: Shared candidate fields; every pass record carries the same pool, so it is kept once.
_SHARED_FIELDS = ("rows", "candidates", "candidates_debug")


def configured_passes(raw: str | None) -> int:
    """Parse the judge pass count.

    Args:
        raw (str | None): ``RUBIKSQL_JUDGE_PASSES``; unset or empty means one pass.

    Returns:
        int: The positive pass count.

    Raises:
        ValueError: The value is not a positive integer.
    """
    if raw in (None, ""):
        return 1
    try:
        passes = int(raw)
    except ValueError:
        raise ValueError(f"{PASSES_ENV} must be a positive integer, got {raw!r}") from None
    if passes < 1:
        raise ValueError(f"{PASSES_ENV} must be a positive integer, got {raw!r}")
    return passes


def answered(result: object) -> bool:
    """Whether one pass ended with an executed final answer.

    Args:
        result (object): The pass's final vote result.

    Returns:
        bool: True for a mapping with non-empty SQL and no execution error.
    """
    return isinstance(result, Mapping) and not result.get("error") and bool((result.get("sql") or "").strip())


def majority(results: Sequence[object], result_sig: Callable[[Mapping[str, Any]], Hashable]) -> tuple[int, list[list[int]]]:
    """Choose one pass by result-equivalence majority.

    Args:
        results (Sequence[object]): Final results in pass order. An unanswered result
            (see `answered`) belongs to a group of its own.
        result_sig (Callable[[Mapping[str, Any]], Hashable]): Official execution-result signature.

    Returns:
        tuple[int, list[list[int]]]: The chosen 0-based pass, and every group as 1-based
            pass numbers ordered by descending size and then by earliest member.

    Raises:
        ValueError: ``results`` is empty.
    """
    if not results:
        raise ValueError("a judge majority needs at least one pass")
    groups: dict[Hashable, list[int]] = {}
    for index, result in enumerate(results):
        key = ("rows", result_sig(result)) if answered(result) else ("failure", index)
        groups.setdefault(key, []).append(index)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
    return ordered[0][0], [[index + 1 for index in members] for members in ordered]


def judge_source(result: object) -> str | None:
    """Return the immutable source id of a pass's judge record, else ``None``.

    Args:
        result (object): The pass's final vote result.

    Returns:
        str | None: ``judge_debug.cache_source.id``, the id independence checks count.
    """
    debug = result.get("judge_debug") if isinstance(result, Mapping) else None
    source = debug.get("cache_source") if isinstance(debug, Mapping) else None
    return source.get("id") if isinstance(source, Mapping) else None


def pass_record(index: int, result: Mapping[str, Any]) -> dict[str, Any]:
    """Project one pass's final result into the question record.

    Args:
        index (int): The 0-based pass index.
        result (Mapping[str, Any]): The pass's final vote result.

    Returns:
        dict[str, Any]: ``pass`` (1-based) plus every result field except the rows and
            the shared candidate pool, so its judge, reviewer and hook records stay whole.
    """
    return {"pass": index + 1, **{key: value for key, value in result.items() if key not in _SHARED_FIELDS}}


def decide(results: Sequence[Mapping[str, Any]], result_sig: Callable[[Mapping[str, Any]], Hashable], *,
           question: str = "") -> dict[str, Any]:
    """Return the majority pass's result carrying every pass and the decision.

    Args:
        results (Sequence[Mapping[str, Any]]): Final results of the completed passes, in pass order.
        result_sig (Callable[[Mapping[str, Any]], Hashable]): Official execution-result signature.
        question (str): Canonical request for optional rank-contract publication.

    Returns:
        dict[str, Any]: The chosen pass's result plus ``judge_passes`` (see `pass_record`)
            and ``judge_majority`` (``passes``, 1-based ``chosen_pass``, ``groups`` and
            ``judge_sources``, as the offline majority file records them). A pass that
            consumed a failed retrieval marks the whole answer degraded.
    """
    chosen, groups = majority(results, result_sig)
    from .sql_action import publish_rank_result, rank_publication_enabled

    order = [i - 1 for group in groups for i in group]
    published = publish_rank_result(question, dict(results[chosen]),
                                    [dict(results[i]) for i in order],
                                    enabled=rank_publication_enabled())
    if (published.get("rank_contract") or {}).get("fired"):
        chosen = order[published["rank_contract"]["selected_alternative"]]
    decided = {**published,
               "judge_passes": [pass_record(index, result) for index, result in enumerate(results)],
               "judge_majority": {"v": 1, "implementation_version": IMPLEMENTATION_VERSION,
                                  "passes": len(results), "chosen_pass": chosen + 1, "groups": groups,
                                  "judge_sources": [judge_source(result) for result in results]}}
    if any(result.get("degraded_retrieval") for result in results):
        decided["degraded_retrieval"] = True
    return decided
