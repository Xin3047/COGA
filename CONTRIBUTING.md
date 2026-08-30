# Contributing

Thank you for helping make COGA easier to falsify and reproduce.

1. Open an issue before a large method, schema, or benchmark change.
2. Create a focused branch and keep generated data/model artifacts out of Git.
3. Add tests for deterministic transforms, data safety, and failure behavior.
4. Run `ruff check src tests scripts` and `pytest -q`.
5. Describe the exact config, source revision, hardware, and commands in the pull request.

Scientific changes must preserve the separation between selection data and benchmark results.
Never tune selection on held-out benchmark outcomes, silently convert censored examples to
failures, weaken a gate after observing results, or add privileged fields to student input.

By contributing, you agree that your contribution is licensed under Apache-2.0.
