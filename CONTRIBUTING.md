# Contributing

KPM-Bridge welcomes focused fixes that preserve the distinction between
protocol interoperability, semantic portability, and downstream xApp safety.

## Before opening a change

1. Open an issue for changes that alter the canonical contract, certificate
   schema, benchmark split, or manuscript claims.
2. Keep raw ColO-RAN traces outside Git; use `make fetch` and the pinned
   manifest instead.
3. Do not replace a deterministic seed or a benchmark output without explaining
   the scientific reason and regenerating every dependent artifact.

## Local validation

```bash
python3 -m pip install -e '.[test]'
make test
make smoke
```

For changes to claim-bearing code or manuscript evidence, also run the full
protocol in `REPRODUCIBILITY.md`. Pull requests should state which commands
were run and whether any result changed.

## Style and scope

- Prefer small, typed functions and deterministic tests.
- Preserve fail-closed behavior for stale, unsupported, or drifted inputs.
- Keep commercial-vendor claims out of controlled implementation profiles.
- Update `CITATION.cff`, documentation, and the changelog when preparing a
  release.

By contributing, you agree that code contributions are provided under the MIT
License. Manuscript text and publication figures require explicit author
approval before redistribution or relicensing.
