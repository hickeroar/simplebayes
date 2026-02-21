from fastapi import FastAPI

from simplebayes import SimpleBayes
from simplebayes.api.routes import create_router


def create_app(auth_token: str = "") -> FastAPI:
    classifier = SimpleBayes()
    app = FastAPI(title="simplebayes")
    app.include_router(create_router(classifier, auth_token=auth_token))
    return app


app = create_app()
