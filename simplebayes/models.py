from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClassificationResult:
    """Structured classification output."""

    category: Optional[str]
    score: float


@dataclass(frozen=True)
class CategorySummary:
    """Summary values for one trained category."""

    token_tally: int
    prob_in_cat: float
    prob_not_in_cat: float
