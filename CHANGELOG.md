# simplebayes Changelog

All notable changes to this project are documented here.

## v3.0.0 - Unreleased

### Breaking
- Introduced a full HTTP API runtime with CLI entrypoint and expanded package layout.
- Added typed classification/result contracts and stricter category validation semantics.
- Added versioned JSON model persistence APIs (`save`/`load`, `save_to_file`/`load_from_file`) with validation and atomic file writes.

### Added
- FastAPI API surface:
  - `/info`
  - `/train/{category}`
  - `/untrain/{category}`
  - `/classify`
  - `/score`
  - `/flush`
  - `/healthz`
  - `/readyz`
- Optional bearer token protection for all non-probe endpoints.
- 1 MiB request body guardrails for text endpoints.
- Readiness lifecycle state with drain behavior on shutdown.
- CLI runner (`simplebayes-server`) with host/port/auth-token options and env var support.
- Dedicated tokenizer pipeline with Unicode normalization, lowercasing, non-word splitting, and English stemming.
- Thread-safe classifier state handling and concurrency stress tests.
- New docs set:
  - rewritten README with API and operational guidance
  - CONTRIBUTING guide

### Changed
- CI now includes expanded quality lanes:
  - strict 100% coverage gate
  - API integration slice
  - packaging/build smoke lane
  - scheduled/manual workflow triggers

## v2.1.0

### Added
- Release bump to 2.1.0.

### Changed
- Cache path handling and docs cleanup improvements.
- Repository/license housekeeping updates.

## v2.0.0

### Added
- Modernized project/tooling baseline for current Python versions.
- Added PyPI publish workflow.
- Added `.flake8` handling updates for test/lint compatibility.

### Changed
- Refreshed release and packaging workflow.

## v1.5.8

### Changed
- No code delta from v1.5.7 (tag alignment release marker).

## v1.5.7

### Added
- Expanded docs generation and API documentation links.

### Fixed
- Critical scoring behavior after cache reload.
- Miscellaneous documentation and docstring fixes.

### Changed
- README and docs hosting updates.

## v1.5.5

### Changed
- Updated licensing metadata and README license coverage.

## v1.5.4

### Changed
- Reverted project URL configuration update.

## v1.5.3

### Changed
- Renamed internal function(s) for readability.
- Version metadata bump.

## v1.5.2

### Added
- Added distribution script and moved packaging flow to setuptools.

## v1.5.1

### Added
- Initial classifier core with train/untrain/score/classify behavior.
- Basic packaging and setup metadata.
- Persistence support and accompanying unit tests.
- Full test coverage wiring and early build checks.

### Changed
- Bayes scoring approach rewrite and performance optimizations.
- Python 2/3 compatibility updates during early lifecycle.
- Documentation, README, and homepage iterations.
