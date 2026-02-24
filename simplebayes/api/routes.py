import sys
import secrets
from typing import Dict

from fastapi import APIRouter, Body, Depends, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from simplebayes import SimpleBayes
from simplebayes.errors import UnauthorizedError
from simplebayes.runtime.readiness import ReadinessState
from simplebayes.api.schemas import (
    CategorySummaryResponse,
    ClassificationResponse,
    InfoResponse,
    MutationResponse,
)

CATEGORY_REGEX = r"^[-_A-Za-z0-9]{1,64}$"


def _get_classifier(request: Request) -> SimpleBayes:
    return request.app.state.classifier


def _get_readiness(request: Request) -> ReadinessState:
    return request.app.state.readiness


def _log_verbose(request: Request, *parts: str) -> None:
    """Log to stderr when verbose mode is enabled."""
    if getattr(request.app.state, "verbose", False):
        print("[simplebayes]", *parts, file=sys.stderr)


def _format_tokens(tokens: list) -> str:
    """Format token list, truncating if long."""
    max_show = 20
    if len(tokens) <= max_show:
        return str(tokens)
    return str(tokens[:max_show]) + "..."


MAX_REQUEST_BODY_BYTES = 1024 * 1024
WWW_AUTH_HEADER = {"WWW-Authenticate": 'Bearer realm="simplebayes"'}


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


def _create_auth_dependency(auth_token: str):
    """Returns a FastAPI dependency for Bearer auth. When auth_token is empty, no auth."""
    bearer = HTTPBearer(auto_error=False)

    def verify(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if not auth_token:
            return
        if credentials is None:
            raise UnauthorizedError()
        if not secrets.compare_digest(credentials.credentials, auth_token):
            raise UnauthorizedError()

    return verify


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


def create_router(auth_token: str = "", verbose: bool = False) -> APIRouter:
    router = APIRouter()
    verify_auth = _create_auth_dependency(auth_token)

    @router.get("/info", response_model=InfoResponse)
    def info(
        request: Request,
        _auth: None = Depends(verify_auth),
        classifier: SimpleBayes = Depends(_get_classifier),
    ):
        result = InfoResponse(categories=_map_summaries(classifier))
        _log_verbose(
            request,
            "info:",
            "categories=",
            str(list(result.categories.keys())),
        )
        return result

    @router.post("/train/{category}", response_model=MutationResponse)
    def train(
        request: Request,
        _auth: None = Depends(verify_auth),
        classifier: SimpleBayes = Depends(_get_classifier),
        category: str = Path(..., pattern=CATEGORY_REGEX),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        tokens = classifier.tokenizer(text)
        classifier.train(category, text)
        summaries = _map_summaries(classifier)
        _log_verbose(
            request,
            "train:",
            "category=",
            category,
            "tokens=",
            _format_tokens(tokens),
            "summaries=",
            str({k: v.tokenTally for k, v in summaries.items()}),
        )
        return MutationResponse(success=True, categories=summaries)

    @router.post("/untrain/{category}", response_model=MutationResponse)
    def untrain(
        request: Request,
        _auth: None = Depends(verify_auth),
        classifier: SimpleBayes = Depends(_get_classifier),
        category: str = Path(..., pattern=CATEGORY_REGEX),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        tokens = classifier.tokenizer(text)
        classifier.untrain(category, text)
        summaries = _map_summaries(classifier)
        _log_verbose(
            request,
            "untrain:",
            "category=",
            category,
            "tokens=",
            _format_tokens(tokens),
            "summaries=",
            str({k: v.tokenTally for k, v in summaries.items()}),
        )
        return MutationResponse(success=True, categories=summaries)

    @router.post("/classify", response_model=ClassificationResponse)
    def classify(
        request: Request,
        _auth: None = Depends(verify_auth),
        classifier: SimpleBayes = Depends(_get_classifier),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        tokens = classifier.tokenizer(text)
        result = classifier.classify_result(text)
        _log_verbose(
            request,
            "classify:",
            "tokens=",
            _format_tokens(tokens),
            "category=",
            str(result.category),
            "score=",
            str(result.score),
        )
        return ClassificationResponse(category=result.category, score=result.score)

    @router.post("/score")
    def score(
        request: Request,
        _auth: None = Depends(verify_auth),
        classifier: SimpleBayes = Depends(_get_classifier),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        text, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        tokens = classifier.tokenizer(text)
        scores = classifier.score(text)
        _log_verbose(
            request,
            "score:",
            "tokens=",
            _format_tokens(tokens),
            "scores=",
            str(scores),
        )
        return scores

    @router.post("/flush", response_model=MutationResponse)
    def flush(
        request: Request,
        _auth: None = Depends(verify_auth),
        classifier: SimpleBayes = Depends(_get_classifier),
        payload: bytes = Body(b"", media_type="text/plain"),
    ):
        _, payload_response = _parse_payload(payload)
        if payload_response is not None:
            return payload_response

        classifier.flush()
        _log_verbose(request, "flush: Flushed all categories")
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    def readyz(readiness: ReadinessState = Depends(_get_readiness)):
        if readiness.is_ready:
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not ready"})

    return router
