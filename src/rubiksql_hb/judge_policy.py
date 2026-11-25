"""Evaluation model policy, with an explicit uncounted diagnostic exception.

The judge of a run is its pinned release's ``judge_model`` knob (``prompts/entries.py``):
``qwen:qwen3.7-max`` unless the release selects ``qwen:qwen3.8-max``. A non-diagnostic run uses
exactly that value; the reviewer follows the judge. Diagnostic entry points may choose either
allowed judge and additionally verify the chosen models against their committed paid admission
before starting workers.
"""
from __future__ import annotations

from .prompts.entries import KNOBS
from .experiment_gateway import DEEPSEEK_CANDIDATE_MODEL

__all__ = ["JUDGE", "JUDGES", "require_judge", "require_diagnostic_configuration", "require_candidate_mode"]

#: The judge of a release that does not set ``judge_model``, as a ``provider:model`` spec.
JUDGE: str = KNOBS["judge_model"].default
#: Every judge a release may select and a diagnostic may choose, as ``provider:model`` specs.
JUDGES: tuple[str, ...] = KNOBS["judge_model"].choices
DIAGNOSTIC_CANDIDATES = frozenset(("qwen:qwen3.7-plus", "qwen:qwen3.8-max", f"deepseek:{DEEPSEEK_CANDIDATE_MODEL}"))


def require_judge(spec: str, *, diagnostic: bool = False, release: str | None = JUDGE) -> str:
    """Return a permitted judge: an allowed one and, outside diagnostic mode, the pinned release's.

    Args:
        spec (str): A ``provider:model`` judge spec, as ``--judge`` takes it.
        diagnostic (bool): Permit any allowed judge; paid admission must independently verify the
            diagnostic mode and configuration.
        release (str | None): The pinned release's ``judge_model`` (``Release.knob("judge_model")``),
            which a non-diagnostic judge must equal. Defaults to ``JUDGE``, the value of a release
            that does not set the knob. ``None`` is for a launcher that has not pinned its release
            yet: only the allowed set is checked, and its pinned workers check the rest.

    Returns:
        str: ``spec``, unchanged.

    Raises:
        ValueError: If ``spec`` is not an allowed judge, or a non-diagnostic ``spec`` differs from
            ``release``; the message names it.

    Examples:
        >>> require_judge("qwen:qwen3.7-max")
        'qwen:qwen3.7-max'
        >>> require_judge("qwen:qwen3.8-max", release="qwen:qwen3.8-max")
        'qwen:qwen3.8-max'
    """
    if spec not in JUDGES:
        raise ValueError(f"the judge is {' or '.join(JUDGES)} only; got {spec!r}")
    if not diagnostic and release is not None and spec != release:
        raise ValueError(f"a non-diagnostic run's judge is only its pinned release's judge_model {release}; got {spec!r}")
    return spec


def require_diagnostic_configuration(candidates: str, judge: str, *, saved_candidates: bool = False) -> None:
    """Validate the model choices of a descriptive measurement before admission.

    Args:
        candidates: Ordered comma-separated ``provider:model:count`` specs, fresh or saved.
        judge: The judge and reviewer's ``provider:model`` spec.
        saved_candidates: Whether the caller uses a separately validated saved pool.
            Both sources share this model policy; admission verifies saved provenance.

    Returns:
        None: The specifications satisfy the diagnostic model policy.

    Raises:
        ValueError: Unsupported model, malformed spec or nonpositive count.
    """
    for entry in candidates.split(","):
        parts = entry.split(":")
        if (len(parts) != 3 or ":".join(parts[:2]) not in DIAGNOSTIC_CANDIDATES
                or not parts[2].isascii() or not parts[2].isdecimal() or int(parts[2]) <= 0):
            raise ValueError("diagnostic candidates require one of " + ", ".join(sorted(DIAGNOSTIC_CANDIDATES))
                             + " with a positive integer count")
    require_judge(judge, diagnostic=True)


def require_candidate_mode(spec: str | None, *, diagnostic: bool = False) -> None:
    """Accept the adopted Qwen 3.8 Max candidates in production as in the experiment kit.

    Args:
        spec: Candidate specs accepted by the evaluator, or no vote candidates.
        diagnostic: Whether the caller will verify a diagnostic paid admission.

    Raises:
        ValueError: An unadopted Qwen 3.8 model is selected outside diagnostic mode.
    """
    if diagnostic:
        return
    for entry in (spec or "").split(","):
        parts = entry.strip().split(":")
        if len(parts) >= 2 and parts[1].startswith("qwen3.8") and parts[1] != "qwen3.8-max":
            raise ValueError("Only the adopted qwen3.8-max candidate is supported outside diagnostics")
