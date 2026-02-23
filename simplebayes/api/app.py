from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from simplebayes import SimpleBayes
from simplebayes.api.routes import WWW_AUTH_HEADER, create_router
from simplebayes.errors import UnauthorizedError
from simplebayes.runtime.readiness import ReadinessState


def create_app(auth_token: str = "") -> FastAPI:
    classifier = SimpleBayes()

    readiness = ReadinessState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        readiness.mark_ready()
        yield
        readiness.mark_not_ready()

    app = FastAPI(title="simplebayes", lifespan=lifespan)
    app.state.classifier = classifier
    app.state.readiness = readiness
    app.include_router(create_router(auth_token=auth_token))

    @app.exception_handler(UnauthorizedError)
    def unauthorized_handler(_request: Request, _exc: UnauthorizedError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized"},
            headers=WWW_AUTH_HEADER,
        )

    return app


app = create_app()
