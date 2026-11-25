"""Scoring metrics for BIRD eval.

EX lives in :mod:`.official_ex` and IS the official BIRD comparison —
``set(predicted_res) == set(ground_truth_res)`` over raw positional tuples. It
is not a local reimplementation: the vendored ``ex_score.py`` (a RubikBench
port with its own row encoding and dedup/order knobs) was deleted outright so
no second EX with potential subtle divergence can exist. Every EX consumer —
eval_bird, lab_score, lab_probe, scripts/rescore_ex.py — imports
``official_ex_match`` from here.

BF/SF are vendored from RubikBench@730aa3a420948e8cab695f42c3412a19acf4d36f
(src/rubikbench/metrics/{bf_score,sf_score,utils}.py, Apache-2.0) so the eval
scoring chain no longer imports rubikbench → agent-heaven (ahvn). Still
byte-for-byte upstream: ``bf_score.py`` and ``sf_score.py``; ``utils`` is
locally patched only in ``strip_sql_comments`` (inlined from ahvn instead of
imported). They normalize with ``ndigits=3`` and play no part in EX.

Keep these implementations stable: EX/BF/SF numbers must stay comparable across runs.
"""
from .bf_score import bfbeta_score
from .official_ex import official_ex_match, row_tuple
from .sf_score import soft_fbeta_score

__all__ = ["bfbeta_score", "official_ex_match", "row_tuple", "soft_fbeta_score"]
