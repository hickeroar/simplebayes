from typing import Dict

from fastapi import APIRouter, Body, Path

from simplebayes import SimpleBayes
from simplebayes.api.schemas import (
    CategorySummaryResponse,
    ClassificationResponse,
    InfoResponse,
    MutationResponse,
)

CATEGORY_REGEX = r"^[-_A-Za-z0-9]{1,64}$"


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


def create_router(classifier: SimpleBayes) -> APIRouter:
    router = APIRouter()

    def parse_body(payload: bytes) -> str:
        return payload.decode("utf-8", errors="ignore")

    @router.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        return InfoResponse(categories=_map_summaries(classifier))

    @router.post("/train/{category}", response_model=MutationResponse)
    def train(
        category: str = Path(..., pattern=CATEGORY_REGEX),
        payload: bytes = Body(b"", media_type="text/plain"),
    ) -> MutationResponse:
        classifier.train(category, parse_body(payload))
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.post("/untrain/{category}", response_model=MutationResponse)
    def untrain(
        category: str = Path(..., pattern=CATEGORY_REGEX),
        payload: bytes = Body(b"", media_type="text/plain"),
    ) -> MutationResponse:
        classifier.untrain(category, parse_body(payload))
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.post("/classify", response_model=ClassificationResponse)
    def classify(payload: bytes = Body(b"", media_type="text/plain")) -> ClassificationResponse:
        result = classifier.classify_result(parse_body(payload))
        return ClassificationResponse(category=result.category or "", score=result.score)

    @router.post("/score")
    def score(payload: bytes = Body(b"", media_type="text/plain")) -> Dict[str, float]:
        return classifier.score(parse_body(payload))

    @router.post("/flush", response_model=MutationResponse)
    def flush(payload: bytes = Body(b"", media_type="text/plain")) -> MutationResponse:
        _ = payload
        classifier.flush()
        return MutationResponse(success=True, categories=_map_summaries(classifier))

    @router.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    def readyz() -> Dict[str, str]:
        return {"status": "ready"}

    return router
