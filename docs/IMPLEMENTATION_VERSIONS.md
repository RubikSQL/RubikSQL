# Implementation version audit

The public runtime executes the verified release `r7bb8146d8e5290da` from
`releases/adopted-snapshot.tar.gz`. Its 26 selected Tool records load 29 Python
modules through `BundleFinder`. The files under `src/rubiksql_hb/tools/` are
authoring inputs; importing a checkout tool without the verified bundle is
blocked. Keeping different code under those same module names made it easy to
mistake unadopted work for the version users actually run.

## Duplicate implementations found

| Checkout files | Unadopted difference from the executing release |
| --- | --- |
| `expand_examples.py`, `get_examples.py`, `query_knowledge.py`, `query_spec.py` | Example applicability checks and `example_scope` output were added after the selected release. `get_examples.py` also offered inline SQL. |
| `run_sql.py`, `run_sql_base.py`, `submit_final_sql.py` | SQL results were wrapped through `sql_result`; that helper is absent from this public checkout. |
| `example_scope.py` | Helper used only by the unadopted example changes. |
| `answering_tools.json` | The `get_examples` and `expand_examples` descriptions advertised the unadopted example behavior. |

These seven tool implementations and two declarations now match the active
release. The unused `example_scope.py` was removed. The 29 checkout tool
modules have the same UTF-8 bytes as the verified snapshot; `.gitattributes`
keeps their line endings stable across Windows and Unix checkouts. The
original RubikSQL-HB repository still contains the later experimental work.

## Similar names that remain

- The archived snapshot contains 43 Tool records but selects 26 for this
  release. Older records are verification history, not alternate tools loaded
  at runtime. Removing them would change the signed snapshot digest.
- `hb_tool_entities.py`, `tool_bundle.py`, and `prompts/` still recognize older
  bundle or closure formats to verify historical release ancestry. Schema 1
  execution is already refused. These paths are retained because snapshot
  validation still calls them; pruning them would require changing the
  authority and snapshot format.
- `vote.py` uses its v2 risk signal as the current default. The v1 checklist
  is no longer an alternate function. `nl2sql` and `nl2sql_vote` are distinct
  public workflows, not duplicate generations of one function.

No duplicate top-level function or class name was found within a Python file.
This audit did not change the adopted archive, release pointer, importer, or
NL2SQL execution algorithm.
