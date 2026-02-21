class SimpleBayesError(Exception):
    """Base exception for simplebayes domain errors."""


class InvalidCategoryError(SimpleBayesError):
    """Raised when a category value is invalid."""


class PersistencePathError(SimpleBayesError):
    """Raised when a persistence path is invalid."""


class UnsupportedModelVersionError(SimpleBayesError):
    """Raised when a persisted model version cannot be loaded."""


class InvalidModelStateError(SimpleBayesError):
    """Raised when persisted model data is malformed or inconsistent."""


class PayloadTooLargeError(SimpleBayesError):
    """Raised when inbound payload exceeds configured limits."""
