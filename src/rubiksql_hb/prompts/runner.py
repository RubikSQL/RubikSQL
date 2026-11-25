"""The bound evolution workspace adapter and its read-only MCP access."""
from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, TypeVar

if TYPE_CHECKING:
    from .types import AuthorityBinding, GitAuthorityBinding

__all__ = ["LAB_SERVER_NAME", "LAB_TOOLS", "Lab", "clear_analysis_scope",
           "lab_problem", "mcp_config", "scoped_lab_server"]

LAB_SERVER_NAME = "lab"
LAB_TOOLS = ("query_lab", "read_evidence", "run_analysis_sql", "list_entities",
             "describe_entity", "get", "count", "query", "explain")
_Result = TypeVar("_Result")


def _server_environment() -> dict[str, str]:
    """Keep stdio readers on this framework's schema, including worktree launches.

    MCP clients pass only their explicit environment and a small default set.
    Pin this package root before inherited paths so a shared editable install
    cannot substitute another checkout's framework or entity definitions.
    """
    source = str(Path(__file__).resolve().parents[2])
    inherited = os.environ.get("PYTHONPATH")
    return {"RUBIKSQL_TOOL_BUNDLE": "", "RUBIKSQL_PROMPT_SNAPSHOT": "", "RUBIKSQL_PROMPT_RELEASE": "",
            "PYTHONPATH": source + (os.pathsep + inherited if inherited else "")}


class Lab:
    """The lab workspace of a site: the runs ingested, with or without their transcripts, read by the proposer.

    Args:
        directory (Path | None): Bound workspace database or directory assertion.
            Omission uses the ordinary package binding.
        db_dir (Path | None): The SQLite sources the evaluations read; each ingested run's candidates are scored against
            their run's gold from them (:func:`rubiksql_hb.lab_score.score_candidates`). ``None`` skips the scoring, and
            the per-slot accuracy stays unknown.
        python (str): The interpreter the lab server runs with.
        questions_path (Path | None): Exact question set used by ingestion.
        binding (AuthorityBinding | None): Explicit identity for scratch or restored
            workspaces. Normal consumers use the ordinary package binding.
        tool_deployment (str | None): Exact independent META deployment to capture.
            Omission resolves the latest deployment once at server startup.
    """

    def __init__(self, directory: Path | None = None, db_dir: Path | None = None, python: str = sys.executable,
                 *, questions_path: Path | None = None, binding: AuthorityBinding | GitAuthorityBinding | None = None,
                 tool_deployment: str | None = None) -> None:
        from ..lab_schema import lab_binding

        self.binding = lab_binding(directory, binding=binding)
        self._selector = directory
        self.db_dir = Path(db_dir) if db_dir is not None else None
        self.python = python
        self.tool_deployment = tool_deployment
        self.questions_path = questions_path
        self._ws: Any = None
        self._ingestion_depth = 0

    @property
    def path(self) -> Path:
        """Physical database of this object's fixed authority projection."""
        from .types import GitAuthorityBinding

        return (Path(self.open()._rubiksql_authority_db) if isinstance(self.binding, GitAuthorityBinding)
                else self.binding.db_path)

    @property
    def directory(self) -> Path:
        """Private local directory containing the current read projection."""
        return self.path.parent

    def open(self) -> Any:
        """Open the lab without importing questions or any previous experiment.

        Returns:
            HeavenBase: The workspace, with the agent extension enabled.
        """
        if self._ws is None:
            from ..lab_schema import open_lab_workspace

            self._ws = open_lab_workspace(self._selector, binding=self.binding, create=False)
        return self._ws

    def transact(self, operation: Callable[[Lab], _Result]) -> _Result:
        """Revalidate and publish a complete replay-safe lab command after contention.

        Args:
            operation: Command callback that checks its original input identities
                on every attempt and performs no external execution effects.

        Returns:
            Callback result after confirmed publication.

        Raises:
            AuthorityError: Changed inputs, non-conflict failure or exhausted CAS retries.
        """
        from .authority import Authority
        from .types import GitAuthorityBinding

        def attempt(writer: Any) -> _Result:
            previous = self._ws
            self._ws = writer.control.ws
            # Nested ingestion must not close the outer writer's workspace.
            self._ingestion_depth += 1
            try:
                return operation(self)
            finally:
                self._ingestion_depth -= 1
                self._ws = previous

        try:
            return Authority.transact(attempt, binding=self.binding)
        finally:
            if isinstance(self.binding, GitAuthorityBinding) and not self._ingestion_depth:
                if self._ws is not None:
                    self._ws.close()
                self._ws = None
                self._selector = None

    @contextlib.contextmanager
    def ingestion_lock(self, operation: str = "lab-ingestion") -> Iterator[Any]:
        """Hold one bound authority batch, sharing nested writers on this thread.

        Args:
            operation: Domain operation recorded once on successful batch completion.

        Yields:
            The active writer's canonical workspace. Content refresh and operational
            evidence accounting finish before the writer lock is released.
        """
        from ..lab_store import lab_write
        from .types import GitAuthorityBinding

        source = self.open()
        previous = self._ws
        self._ingestion_depth += 1
        try:
            with lab_write(source, operation) as ws:
                self._ws = ws
                yield ws
        finally:
            self._ws = previous
            self._ingestion_depth -= 1
            if isinstance(self.binding, GitAuthorityBinding) and not self._ingestion_depth:
                previous.close()
                self._ws = None
                self._selector = None

    def ingest(self, run_dir: Path, name: str, *, cache_db: Path | None,
               iteration: int | None = None, parent_experiment: str | None = None,
               transcripts: bool = True) -> dict[str, Any]:
        """Ingest and score one verified run under the lab-wide lock (see :meth:`_ingest`)."""
        with self.ingestion_lock("ingest-run"):
            return self._ingest(run_dir, name, cache_db=cache_db, iteration=iteration,
                                parent_experiment=parent_experiment, transcripts=transcripts)

    def _ingest(self, run_dir: Path, name: str, *, cache_db: Path | None, iteration: int | None = None,
                parent_experiment: str | None = None, transcripts: bool = True) -> dict[str, Any]:
        """Ingest one run as experiment ``name``, unless the lab already holds exactly its result files.

        A new run registers its traces, candidates and judges, plus sessions and messages when ``transcripts`` is true.
        The judges' and reviewers' conversations can come from ``cache_db``. A run the lab holds under the same
        result-file digests, snapshot and transcript storage mode is left as it is. Either way its unscored candidates
        are then scored when :attr:`db_dir` is set (a pass an interruption cut short resumes), which refreshes the
        experiment's aggregates.

        Args:
            run_dir (Path): The run directory.
            name (str): The experiment id (the run directory's name, for the loop's runs).
            cache_db (Path | None): The LLM cache the run wrote.
            iteration (int | None): The loop iteration that launched the run.
            parent_experiment (str | None): The experiment a child run is paired with.
            transcripts (bool): Store the run's ``agent-session`` and ``agent-message`` rows. ``False``
                (``meta_ops record``) stores scores, SQL and execution results only and removes sessions an earlier
                ingest of the experiment stored; the run directory keeps the conversations.

        Returns:
            dict[str, Any]: ``{"experiment", "ingested", "session_sources"}``, plus ``lab_ingest.ingest_run``'s counts when
                the run was ingested now, and the candidate scoring counts (``scored``) when :attr:`db_dir` is set.
        """
        from ..lab_ingest import experiment_session_sources, ingest_run, recorded_run_inputs, run_result_files, run_snapshot
        from ..lab_identity import database_path, question_file, validate_database_version
        from ..lab_schema import ExperimentEntity

        snapshot = run_snapshot(run_dir)
        dataset_version, fingerprints, _ = recorded_run_inputs(run_dir)
        from ..evaluation_questions import QUESTION_SETS

        questions_path = self.questions_path
        if questions_path is None:
            selected = next((item for item in QUESTION_SETS.values() if item.digest == dataset_version), None)
            if selected is None:
                raise ValueError(f"{run_dir}: question set is not an allowlisted evaluation input")
            questions_path = Path(__file__).resolve().parents[3] / selected.path
        _, actual_dataset, _ = question_file(questions_path)
        if dataset_version != actual_dataset:
            raise ValueError(f"{run_dir}: question content differs from this lab's fixed question set")
        if self.db_dir is not None:
            for db_id, version in fingerprints.items():
                validate_database_version(database_path(self.db_dir, db_id), version)
        ws = self.open()
        existing = ws.get(name, ExperimentEntity)
        existing = existing if isinstance(existing, dict) or existing is None else existing.to_dict()
        provenance = (existing or {}).get("provenance") or {}
        if (existing is not None and existing.get("result_files") == run_result_files(run_dir)
                and (provenance.get("prompt_snapshot") or {}).get("digest") == snapshot.digest
                and provenance.get("transcripts", "lab") == ("lab" if transcripts else "run_directory")):
            out: dict[str, Any] = {"experiment": name, "ingested": False, "session_sources": experiment_session_sources(ws, name)}
        else:
            out = {"experiment": name, "ingested": True, **ingest_run(
                ws, run_dir, name, questions_path=questions_path, db_fingerprints=fingerprints,
                cache_db=cache_db, iteration=iteration, parent_experiment=parent_experiment,
                transcripts=transcripts)}
        if self.db_dir is not None:
            from ..lab_score import score_candidates

            out["scored"] = score_candidates(ws, self.db_dir, exp_id=name)
        return out

    def record_outcome(self, name: str, *, verdict: str | None, gain_pp: float | None) -> None:
        """Patch the verdict and gain of the outcome that paired experiment ``name`` as a child."""
        from ..lab_ingest import record_outcome

        with self.ingestion_lock("record-outcome") as ws:
            record_outcome(ws, name, verdict=verdict, gain_pp=gain_pp)

    def refresh(self) -> dict[str, Any]:
        """Refresh derived GeneralRule/KnobSet views in one bound writer batch.

        Returns:
            Derived counts, changes, authority revision and elapsed time.
        """
        from ..lab_store import refresh_views
        from .authority import Authority

        with self.ingestion_lock("refresh-views") as ws:
            writer = Authority.active_writer(Path(ws._rubiksql_authority_db))
            if writer is None:
                raise RuntimeError("derived refresh requires the active bound writer")
            with writer.reader() as reader:
                return refresh_views(ws, reader)

    def record_knowledge(self, name: str, root: Path) -> dict[str, Any]:
        """Record the ledger of the knowledge root experiment ``name`` actually read.

        Args:
            name (str): An experiment already ingested into this lab.
            root (Path): Its evaluation workspaces, with their knowledge ledgers.

        Returns:
            dict[str, Any]: The recorded knowledge identity and build fields.
        """
        from ..lab_ingest import record_knowledge

        with self.ingestion_lock("record-knowledge") as ws:
            return record_knowledge(ws, name, root)

    def server(self) -> dict[str, Any]:
        """The MCP stdio server spec serving this lab read-only.

        Returns:
            dict[str, Any]: Interpreter and stdio server arguments naming the bound workspace.
        """
        from .binding import binding_arguments
        args = ["-m", "rubiksql_hb.lab_server", *binding_arguments(),
                "--analysis-cache-dir", str(self.directory.parent / "analysis-cache")]
        commit = getattr(self.open(), "_rubiksql_authority_commit", None)
        if commit is not None:
            args.extend(["--authority-commit", commit])
        else:
            args.extend(["--workspace", str(self.path)])
        if self.tool_deployment is not None:
            args.extend(["--tool-deployment", self.tool_deployment])
        if self.db_dir is not None:
            args.extend(["--db-dir", str(self.db_dir)])
        return {"command": self.python, "args": args,
                "env": _server_environment()}


def scoped_lab_server(server: Mapping[str, Any], scope: str) -> dict[str, Any]:
    """Bind a local or SSH (cloud or Windows) lab specification to one agent invocation.

    Args:
        server: A specification produced by this module, CloudRunner or WindowsRunner.
        scope: A caller-generated SHA-256 identifier, never agent-provided input.

    Returns:
        A copied specification with a stable diagnostic scope across MCP restarts.

    Raises:
        ValueError: The scope is not a lowercase SHA-256 identifier.
    """
    if re.fullmatch(r"[0-9a-f]{64}", scope) is None:
        raise ValueError("analysis scope must be a SHA-256 identifier")
    result = dict(server)
    args = list(result.get("args") or [])
    if Path(str(result["command"])).name in ("ssh", "ssh.exe"):
        args[-1] += f" --analysis-scope {scope}"
    else:
        args.extend(["--analysis-scope", scope])
    result["args"] = args
    return result


def clear_analysis_scope(server: Mapping[str, Any]) -> None:
    """Delete invocation SQL caches after the caller archives received evidence.

    Args:
        server: The scoped local or SSH server specification used by that agent.

    Raises:
        ValueError: The specification is unscoped or the server refuses cleanup.
        OSError: The command cannot start.
        subprocess.TimeoutExpired: Cleanup does not finish within 60 seconds.
    """
    args = list(server.get("args") or [])
    if Path(str(server["command"])).name in ("ssh", "ssh.exe"):
        if not args or " --analysis-scope " not in args[-1]:
            raise ValueError("cannot clear an unscoped analysis server")
        args[-1] = args[-1].replace(" --read-only", "") + " --clear-analysis-scope"
    else:
        if "--analysis-scope" not in args:
            raise ValueError("cannot clear an unscoped analysis server")
        args = [value for value in args if value != "--read-only"] + ["--clear-analysis-scope"]
    completed = subprocess.run([str(server["command"]), *args],
                               env={**os.environ, **dict(server.get("env") or {})},
                               stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=60, check=False)
    if completed.returncode:
        raise ValueError(f"analysis scope cleanup failed ({completed.returncode}): {completed.stderr[-1000:]}")


def _tool_rows(reply: Mapping[str, Any]) -> Any:
    """What a ``tools/call`` reply returned: its structured result, else the JSON of its first text block."""
    result = reply.get("result") if isinstance(reply.get("result"), Mapping) else {}
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping) and "result" in structured:
        return structured["result"]
    for block in result.get("content") or []:
        if isinstance(block, Mapping) and block.get("type") == "text":
            try:
                return json.loads(str(block.get("text") or ""))
            except ValueError:
                return block.get("text")
    return None


def lab_problem(server: Mapping[str, Any], experiment: str | None, *, log: Path, timeout: float,
                evidence: dict[tuple[str, str], dict[str, Any]] | None = None) -> str:
    """Why the proposer could not read the lab through ``server``, or ``""``: a read-only evidence access check.

    Starts the server exactly as the proposer's MCP configuration does (:func:`mcp_config`) and speaks MCP over its stdio:
    ``initialize``; ``tools/list``, which must include the initial :data:`LAB_TOOLS` acceptance inventory.
    The verified independent deployment owns discovery and can add source-pinned readers; then ``query_lab`` for
    ``experiment``, which must return it (the server must honor its bound ``--workspace`` selector). The server is stopped
    afterwards, and its standard error kept in ``log``.

    Args:
        server (Mapping[str, Any]): The lab server spec (:meth:`Lab.server`): ``command``, ``args``, optional ``env``.
        experiment (str | None): The experiment the proposer is sent to read; ``None`` checks the tools alone (a decision
            round of a run in which no iteration read the lab).
        log (Path): Receives the server's standard error.
        timeout (float): Seconds for the whole exchange, the server's start included.
        evidence: Optional source-key mapping filled with independently fetched originals.
            Each initial read opens a fixed snapshot; continuation calls reuse its pages.
            The keys specify required ``(kind, id)`` sources; values are replaced only
            after all pages have consistent identities and a verified full-source hash.

    Returns:
        str: The problem, or ``""`` when the lab answers as the proposer needs it.
    """
    command = [str(server["command"]), *(str(arg) for arg in server.get("args") or [])]
    env = {**os.environ, **{str(key): str(value) for key, value in (server.get("env") or {}).items()}}
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as errors:
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, env=env)
        except OSError as error:
            return f"the lab server command {command[0]!r} does not start: {error}"
        lines: queue.Queue[bytes | None] = queue.Queue()

        def pump() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout

        def send(message: Mapping[str, Any]) -> None:
            proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]

        def call(ident: int, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
            send({"jsonrpc": "2.0", "id": ident, "method": method, "params": dict(params)})
            while True:
                line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
                if line is None:
                    raise EOFError(method)
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # not a JSON-RPC line
                if isinstance(message, Mapping) and message.get("id") == ident:
                    return message

        try:
            answer = call(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "rubiksql-lab-probe", "version": "1"}})
            if "error" in answer:
                return f"the lab server refused initialize: {answer['error']}"
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            listed = call(2, "tools/list", {})
            tools = sorted(str(tool.get("name")) for tool in (listed.get("result") or {}).get("tools") or [] if isinstance(tool, Mapping))
            if not set(LAB_TOOLS).issubset(tools):
                return f"the lab deployment is missing required reader tools: {sorted(set(LAB_TOOLS) - set(tools))}"
            if experiment is None and not evidence:
                return ""
            if experiment is not None:
                reply = call(3, "tools/call", {"name": "query_lab", "arguments": {"entity": "experiment", "where": {"name": {"$eq": experiment}}}})
                rows = _tool_rows(reply)
                if (reply.get("result") or {}).get("isError") or not isinstance(rows, list) or not any(isinstance(row, Mapping) and row.get("name") == experiment for row in rows):
                    return f"the lab holds no experiment {experiment!r}: query_lab returned {json.dumps(rows, default=str)[:400]}"
            if evidence:
                from ..lab_evidence import SCHEMA, assemble_evidence

                request_id = 4
                for kind, source_id in evidence:
                    arguments = {"kind": kind, "id": source_id, "offset": 0}
                    pages = []
                    while arguments is not None:
                        reply = call(request_id, "tools/call", {"name": "read_evidence", "arguments": arguments})
                        request_id += 1
                        page = _tool_rows(reply)
                        if (reply.get("result") or {}).get("isError") or not isinstance(page, dict):
                            raise ValueError(f"cannot read evidence {kind} {source_id}: {page}")
                        if (page.get("schema") != SCHEMA or page.get("kind") != kind or page.get("id") != source_id
                                or page.get("offset") != arguments["offset"] or type(page.get("end")) is not int
                                or page["end"] <= arguments["offset"] or "next" not in page):
                            raise ValueError(f"invalid evidence page {kind} {source_id} offset {arguments['offset']}")
                        next_arguments = page["next"]
                        if next_arguments is not None and next_arguments != {
                            "kind": kind, "id": source_id, "offset": page["end"], "expected_sha256": page.get("sha256")
                        }:
                            raise ValueError(f"invalid evidence continuation {kind} {source_id}")
                        pages.append(page)
                        arguments = next_arguments
                    evidence[(kind, source_id)] = assemble_evidence(pages)
            return ""
        except queue.Empty:
            return f"the lab server did not answer within {timeout:g}s; its log is {log}"
        except (EOFError, OSError, ValueError, KeyError, TypeError) as error:
            return f"the lab server stopped answering ({type(error).__name__}: {error}); its log is {log}"
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            reader.join(timeout=10)
            proc.stdout.close()  # type: ignore[union-attr]


def mcp_config(server: Mapping[str, Any]) -> dict[str, Any]:
    """The MCP configuration that attaches the lab server, as Claude Code's ``--mcp-config`` reads it.

    The server is ``alwaysLoad``, so Claude Code puts the lab's tools in the prompt from the first turn; with tool search
    on, its default against the Anthropic API, it would otherwise defer them until the model searches for them. Print
    mode holds the first turn for every pending ``--mcp-config`` server, ``alwaysLoad`` or not, up to ``MCP_TIMEOUT``
    (``loop.AgentAccess.environment``), and then starts it with or without the lab (both read from Claude Code 2.1.268
    and 2.1.281).

    Args:
        server (Mapping[str, Any]): The lab server spec (:meth:`Lab.server`).

    Returns:
        dict[str, Any]: ``{"mcpServers": {"lab": {"type": "stdio", **server, "alwaysLoad": True}}}``.
    """
    return {"mcpServers": {LAB_SERVER_NAME: {"type": "stdio", **dict(server), "alwaysLoad": True}}}
