"""Pinned implementation of knowledge_errors."""
import sys
import traceback


def _retrieval_failed(what: str, db_id: str, exc: BaseException) -> list:
    """Report a retrieval failure on stderr and return the empty result the callers expect.

    Rules and examples ARE this system's measured contribution, and every retrieval path
    answers a failure with the same empty list it uses for "nothing matched". A misconfigured
    chroma backend, workspace, or embedding endpoint would therefore run a whole hidden-test
    submission with zero knowledge injected and produce plausible SQL the entire way, with no
    error, no log line, and no diagnostic in the output — on a one-shot run that cannot be
    repeated. The empty return is kept (server.py owns the caller-facing contract); only the
    silence is removed.

    ``get_rules`` no longer uses this return value: a live outage produced 200 of these
    stderr lines while every affected ROW still shipped clean, because the empty list is
    also what a healthy zero-match looks like and nothing downstream could tell the two
    apart. Its except-branch answers with :data:`RULES_RETRIEVAL_FAILED_ERROR` instead, so
    the failure travels in the payload; this helper still owns the stderr report.
    """
    print(f"[knowledge] {what} FAILED for db={db_id!r} — returning zero results; the model will "
          f"answer this question with no retrieved knowledge. {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    return []
