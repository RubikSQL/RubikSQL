# Vendored HeavenBase dependency

`heavenbase-0.1.2.3-py3-none-any.whl` is the exact HeavenBase build required by this RubikSQL-HB runtime. SHA-256:

```text
bfe6c37496b9498f6005ec95093cf7b6a0f9e920ee3187e1e0b0b791152b96ff
```

It includes downstream changes for exact vector retrieval, empty ordered results, read-only `code_reduce` rejection, and multiline `LIKE`/`ILIKE`. Several different builds use the same `0.1.2.3` version string, so the version alone is insufficient to identify the required behavior. The original RubikSQL-HB source repository contains the complete build provenance and reproduction notes; this public-core branch carries the locked wheel to make installation reproducible without credentials for the HeavenBase source repository.

The wheel's own metadata declares `License-Expression: MIT` and includes a `LICENSE` file. Before a public release, review upstream attribution and confirm redistribution of this downstream build alongside the license selected for RubikSQL itself.
