import argparse
import os
from typing import Sequence

import uvicorn

from simplebayes.api.app import create_app


def _env_bool(name: str, default: bool) -> bool:
    """Return True when env value is in ('1', 'true', 'yes'), case-insensitive."""
    val = os.getenv(name, "").lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simplebayes API server.")
    parser.add_argument("--host", default=os.getenv("SIMPLEBAYES_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SIMPLEBAYES_PORT", "8000")),
    )
    parser.add_argument(
        "--auth-token",
        default=os.getenv("SIMPLEBAYES_AUTH_TOKEN", ""),
    )
    parser.add_argument(
        "--language",
        default=os.getenv("SIMPLEBAYES_LANGUAGE", "english"),
        help="Language code for stemmer and stop words (e.g. english, spanish).",
    )
    parser.add_argument(
        "--remove-stop-words",
        action="store_true",
        default=_env_bool("SIMPLEBAYES_REMOVE_STOP_WORDS", False),
        help="Filter common stop words (the, is, and, etc.).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=_env_bool("SIMPLEBAYES_VERBOSE", False),
        help="Log requests, responses, and classifier operations to stderr.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    app = create_app(
        auth_token=args.auth_token,
        language=args.language,
        remove_stop_words=args.remove_stop_words,
        verbose=args.verbose,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    run()
