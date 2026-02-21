# Contributing

## Local development checks

Run these before opening a PR:

```sh
./.venv/bin/pytest tests/ --cov=simplebayes --cov-fail-under=100 -v
./.venv/bin/flake8 simplebayes tests
./.venv/bin/pylint simplebayes tests --exit-zero
```

Optional but recommended:

```sh
./.venv/bin/pytest tests/test_api_endpoints.py -v
./.venv/bin/pytest tests/test_concurrency.py -v
```

## CI parity

CI should run:
- full tests with 100% coverage gate
- lint checks
- packaging/build validation

When API behavior changes, include endpoint contract tests for status codes and payloads.

## Release and versioning

- Use semantic version tags (for example, `v2.2.0`).
- Keep backward compatibility unless intentionally releasing a breaking change.
- If breaking API or contract changes are introduced, bump the major version before release tagging.
