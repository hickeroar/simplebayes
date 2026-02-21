from simplebayes.errors import (
    InvalidCategoryError,
    InvalidModelStateError,
    PayloadTooLargeError,
    PersistencePathError,
    SimpleBayesError,
    UnsupportedModelVersionError,
)
from simplebayes.models import CategorySummary, ClassificationResult


def test_classification_result_fields():
    result = ClassificationResult(category="spam", score=2.5)
    assert result.category == "spam"
    assert result.score == 2.5


def test_category_summary_fields():
    summary = CategorySummary(token_tally=3, prob_in_cat=0.75, prob_not_in_cat=0.25)
    assert summary.token_tally == 3
    assert summary.prob_in_cat == 0.75
    assert summary.prob_not_in_cat == 0.25


def test_error_hierarchy():
    assert issubclass(InvalidCategoryError, SimpleBayesError)
    assert issubclass(PersistencePathError, SimpleBayesError)
    assert issubclass(UnsupportedModelVersionError, SimpleBayesError)
    assert issubclass(InvalidModelStateError, SimpleBayesError)
    assert issubclass(PayloadTooLargeError, SimpleBayesError)
