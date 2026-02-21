from contextlib import asynccontextmanager

from fastapi import FastAPI

from simplebayes import SimpleBayes
from simplebayes.api.routes import create_router
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
    app.state.readiness = readiness
    app.include_router(create_router(classifier, readiness, auth_token=auth_token))
    return app


app = create_app()
