from typing import Dict

from pydantic import BaseModel


class CategorySummaryResponse(BaseModel):
    tokenTally: int
    probNotInCat: float
    probInCat: float


class InfoResponse(BaseModel):
    categories: Dict[str, CategorySummaryResponse]


class MutationResponse(BaseModel):
    success: bool
    categories: Dict[str, CategorySummaryResponse]


class ClassificationResponse(BaseModel):
    category: str
    score: float
