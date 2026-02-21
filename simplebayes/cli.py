import argparse
import os
from typing import Sequence

import uvicorn

from simplebayes.api.app import create_app


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
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    app = create_app(auth_token=args.auth_token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    run()
