# simplebayes Changelog

All notable changes to this project are documented here.

## v3.2.0

### Added
- CLI options for `simplebayes-server`:
  - `--language` – Language code for stemmer and stop words (default: `english`). Environment: `SIMPLEBAYES_LANGUAGE`.
  - `--remove-stop-words` – Filter common stop words. Environment: `SIMPLEBAYES_REMOVE_STOP_WORDS` (`1`, `true`, `yes` = enabled).
  - `--verbose` – Log requests, responses, and classifier operations to stderr. Environment: `SIMPLEBAYES_VERBOSE` (`1`, `true`, `yes` = enabled).
- Verbose mode: request/response middleware logs method, path, Content-Length, status code, and body preview (truncated at 500 chars) to stderr with `[simplebayes]` prefix.
- Verbose mode: classifier insight for each endpoint – tokens extracted, category operations, scores, and summaries (token lists truncated at 20 items).
- `create_app(language, remove_stop_words, verbose)` – API app factory now accepts classifier and logging options.
- `--help` documents all CLI arguments.

### Changed
- API classifier is now configured from CLI `--language` and `--remove-stop-words` instead of using fixed defaults.
- README: CLI options table, environment variable equivalents, and Verbose mode subsection.

## v3.1.1

### Changed
- Documentation and metadata: updated terminology from "Bayes" to "Bayesian" in PyPI keywords and changelog for consistency with proper nomenclature.

## v3.1.0

### Added
- Classifier options for `SimpleBayes`:
  - `alpha` – Laplace smoothing. Use `0.01` or `1.0` to avoid zero probabilities for tokens unseen in a category; improves handling of sparse vocabularies. Default `0` preserves prior behavior.
  - `language` – Language code for stemmer and stop words (e.g. `"english"`, `"spanish"`, `"french"`). All Snowball languages supported. Default `"english"`.
  - `remove_stop_words` – Filter common stop words when `True`. Default `False` for backwards compatibility.
- Built-in stopword lists for every Snowball language (Arabic, Armenian, Basque, Catalan, Danish, Dutch, English, Esperanto, Estonian, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Irish, Italian, Lithuanian, Nepali, Norwegian, Portuguese, Romanian, Russian, Serbian, Spanish, Swedish, Tamil, Turkish, Yiddish). No download or file storage required.
- `create_tokenizer(language, remove_stop_words)` – factory for language-aware tokenizers.
- API: Bearer auth integrated with OpenAPI docs. `/docs` and `/redoc` expose the Bearer scheme; use the "Authorize" button in Swagger UI for interactive testing.
- `UnauthorizedError` – domain exception for Bearer auth failures; produces 401 with `WWW-Authenticate` header.

### Changed
- Tokenizer pipeline: `language` drives both stemming and stop-word filtering. When `remove_stop_words=True`, stop words are filtered after stemming.
- Laplace smoothing applied in probability calculations when `alpha > 0`.
- API routes refactored to use FastAPI `Depends` for auth, classifier, and readiness state.
- README: Classifier Options table, Tokenization section, Bearer auth docs.
- Pylint: `--fail-under=10` in CONTRIBUTING and README.

## v3.0.0

### Breaking
- Introduced a full HTTP API runtime with CLI entrypoint and expanded package layout.
- Added typed classification/result contracts and stricter category validation semantics.
- Added versioned JSON model persistence APIs (`save`/`load`, `save_to_file`/`load_from_file`) with validation and atomic file writes.
- Removed legacy pickle persistence APIs (`cache_train`, `cache_persist`, `get_cache_location`) in favor of JSON-only persistence.
- Removed legacy `SimpleBayes` constructor cache arguments (`cache_path`, `cache_file`).
- `/classify` now returns `category: null` when no category can be selected.

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
- Bayesian scoring approach rewrite and performance optimizations.
- Python 2/3 compatibility updates during early lifecycle.
- Documentation, README, and homepage iterations.
