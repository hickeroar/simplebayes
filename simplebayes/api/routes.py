import secrets
from typing import Dict

from fastapi import APIRouter, Body, Path, Request
from fastapi.responses import JSONResponse

from simplebayes import SimpleBayes
from simplebayes.api.schemas import (
    CategorySummaryResponse,
    ClassificationResponse,
    InfoResponse,
    MutationResponse,
)
from simplebayes.runtime.readiness import ReadinessState

CATEGORY_REGEX = r"^[-_A-Za-z0-9]{1,64}$"
MAX_REQUEST_BODY_BYTES = 1024 * 1024


def _map_summaries(classifier: SimpleBayes) -> Dict[str, CategorySummaryResponse]:
    summaries = classifier.get_summaries()
    return {
        category: CategorySummaryResponse(
            tokenTally=summary.token_tally,
            probNotInCat=summary.prob_not_in_cat,
            probInCat=summary.prob_in_cat,
        )
        for category, summary in summaries.items()
    }


def _authorize(request: Request, auth_token: str) -> JSONResponse | None:
    if not auth_token:
        return None

    auth_header = request.headers.get("Authorization", "")
    scheme, separator, token = auth_header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized"},
            headers={"WWW-Authenticate": 'Bearer realm="simplebayes"'},
        )

    if not secrets.compare_digest(token, auth_token):
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized"},
            headers={"WWW-Authenticate": 'Bearer realm="simplebayes"'},
        )

    return None


def _parse_payload(payload: bytes) -> tuple[str, JSONResponse | None]:
    if len(payload) > MAX_REQUEST_BODY_BYTES:
        return "", JSONResponse(
            status_code=413,
            content={"error": "request body too large"},
        )

    try:
        return payload.decode("utf-8"), None
    except UnicodeDecodeError:
        return "", JSONResponse(
            status_code=400,
            content={"error": "invalid utf-8 payload"},
        )


def create_router(
    classifier: SimpleBayes,
    readiness: ReadinessState,
    auth_token: str = "",
) -> APIRouter:
    router = APIRouter()

    @router.get("/info", response_model=InfoResponse)
    def info(request: Request):
        auth_response = _authorize(request, auth_token)
        if auth_response is not None:
            return auth_response
        return InfoResponse(categories=_map_summaries(classifier))

    @router.post("/train/{category}", response_model=MutationResponse)
    def train(
        request: Request,
        category: str = Path(..., pattern=CATEGORY_REGEX),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        auth_response = _authorize(request, auth_token)
        if auth_response is not None:
            return auth_response

        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        classifier.train(category, text)
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.post("/untrain/{category}", response_model=MutationResponse)
    def untrain(
        request: Request,
        category: str = Path(..., pattern=CATEGORY_REGEX),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        auth_response = _authorize(request, auth_token)
        if auth_response is not None:
            return auth_response

        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        classifier.untrain(category, text)
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.post("/classify", response_model=ClassificationResponse)
    def classify(request: Request, payload: bytes = Body(b"", media_type="text/plain")):
        auth_response = _authorize(request, auth_token)
        if auth_response is not None:
            return auth_response

        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        result = classifier.classify_result(text)
        return ClassificationResponse(category=result.category, score=result.score)

    @router.post("/score")
    def score(request: Request, payload: bytes = Body(b"", media_type="text/plain")):
        auth_response = _authorize(request, auth_token)
        if auth_response is not None:
            return auth_response

        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        return classifier.score(text)

    @router.post("/flush", response_model=MutationResponse)
    def flush(request: Request, payload: bytes = Body(b"", media_type="text/plain")):
        auth_response = _authorize(request, auth_token)
        if auth_response is not None:
            return auth_response

        _, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        classifier.flush()
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    def readyz():
        if readiness.is_ready:
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not ready"})

    return router
