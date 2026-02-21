from fastapi import FastAPI

from simplebayes import SimpleBayes
from simplebayes.api.routes import create_router


def create_app() -> FastAPI:
    classifier = SimpleBayes()
    app = FastAPI(title="simplebayes")
    app.include_router(create_router(classifier))
    return app


app = create_app()
