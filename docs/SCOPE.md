# NL2SQL source scope

Source baseline: local RubikSQL-HB `main` at `cb41144ecfab1cdef5de56e46898012c3937e2a5`. The former introduction-only RubikSQL `main` is preserved at `archive/legacy-introduction-20261125` (`7a0abeb`); its demo video remains in the current tree.

## Included

- The existing `rubiksql_hb.nl2sql` and `rubiksql_hb.vote` answer paths, MCP server, schema registration, knowledge storage/retrieval, SQL guards, model transport, candidate/judge behavior, and their directly needed support modules.
- The 26 answering tool declarations and their source modules. `answering_tools.json` has been reduced from 38 declarations to those 26; the 12 assistant/META declarations and their dedicated tool modules are omitted.
- The adopted execution snapshot and its release pointer. This snapshot still contains its own verified Python tool source and prompt content. The checkout's tool modules mirror the selected snapshot version, but are not substituted for the pinned snapshot during execution. [Implementation version audit](IMPLEMENTATION_VERSIONS.md) records the cleanup.
- The pinned HeavenBase wheel and dependency lock, a source-checkout MCP launcher, and an offline runtime smoke test.

## Excluded

- BIRD submission and scoring scripts, self-evolution launchers, experiment dashboards, replay/analysis programs, and the old development-document collection.
- The prompt-authoring `prompts/store/` history, recorded self-evolution runs, BIRD question and gold-answer resources, training corpora, and benchmark fixtures.
- Standalone lab/analysis UI modules and META tool implementations not selected by the adopted answering snapshot.

The package layout, pinned tool importer, snapshot verification, and NL2SQL algorithms have **not** been redesigned. Some internal authority, evolution, and lab modules remain in `src/rubiksql_hb` because current runtime and snapshot identity code imports them. They are compatibility dependencies, not exposed self-evolution commands. Removing every such module would require splitting the snapshot/identity layer, which is outside this preparation's instruction to keep the architecture.

## Current verification

`uv sync --offline --frozen --no-dev` completed with the locked dependency set. `tests/smoke_runtime.py` verified the adopted snapshot, imported `nl2sql` and `nl2sql_vote`, loaded 26 pinned tools, ran a tool call, and initialized 42 MCP tools without a model key. This is a startup check, not an end-to-end answer-quality test.

The documentation demo was also run from a separate local clone with a new `.venv`: `uv sync --offline --frozen`, the no-key smoke test, and `examples/demo/demo.py` with a DashScope embedding key all completed. The demo registered two tables and returned the expected rows (Ada 100, Lin 25) through the pinned SQL tool. The optional model-generated `--ask` path was not exercised because this isolated run had no chat gateway pool.

## Follow-ups before a packaged release

1. Select a RubikSQL license; this repository currently has no `LICENSE` file. Confirm attribution/redistribution of the patched HeavenBase wheel (its metadata says MIT).
2. The adopted snapshot includes prompt text and 29 Python tool source files. Account for those files when choosing the public license and package contents.
3. Decide whether to replace the historical authority binding that names the original private source remote. The local snapshot runner does not contact it; authority authoring is not part of this repository.
4. Run an end-to-end query against a disposable database with approved model credentials. The smoke check intentionally makes no paid model request.
5. Review the remaining internal compatibility modules if the desired scope is stricter than “no self-evolution workflow.” A stricter cut needs a separate runtime/authoring boundary change.
