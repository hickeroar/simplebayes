import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from simplebayes import SimpleBayes
from simplebayes.api.routes import WWW_AUTH_HEADER, create_router
from simplebayes.errors import UnauthorizedError
from simplebayes.runtime.readiness import ReadinessState


def create_app(
    auth_token: str = "",
    language: str = "english",
    remove_stop_words: bool = False,
    verbose: bool = False,
) -> FastAPI:
    classifier = SimpleBayes(language=language, remove_stop_words=remove_stop_words)

    readiness = ReadinessState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        readiness.mark_ready()
        yield
        readiness.mark_not_ready()

    app = FastAPI(title="simplebayes", lifespan=lifespan)
    app.state.classifier = classifier
    app.state.readiness = readiness
    app.state.verbose = verbose
    app.include_router(create_router(auth_token=auth_token, verbose=verbose))

    @app.middleware("http")
    async def verbose_middleware(request: Request, call_next):
        if not request.app.state.verbose:
            return await call_next(request)
        method = request.method
        path = request.url.path
        content_length = request.headers.get("content-length", "")
        parts = [f"[simplebayes] {method} {path}"]
        if content_length:
            parts.append(f" (Content-Length: {content_length})")
        print("".join(parts), file=sys.stderr)
        try:
            response = await call_next(request)
        except Exception:
            print("[simplebayes] -> (exception)", file=sys.stderr)
            raise
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        preview_len = 500
        if len(body) > preview_len:
            body_preview = body[:preview_len].decode("utf-8", errors="replace") + "..."
        else:
            body_preview = body.decode("utf-8", errors="replace") if body else ""
        print(f"[simplebayes] -> {response.status_code} {body_preview!r}", file=sys.stderr)
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    @app.exception_handler(UnauthorizedError)
    def unauthorized_handler(_request: Request, _exc: UnauthorizedError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized"},
            headers=WWW_AUTH_HEADER,
        )

    return app


app = create_app()
