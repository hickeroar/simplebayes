import re
import unicodedata
from typing import Callable, List, Set

import snowballstemmer

from simplebayes.stopwords_data import _BUILTIN_STOPWORDS

TOKEN_SPLIT_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
_STOPWORDS_CACHE: dict[str, Set[str]] = {}


def _get_stop_words(language: str) -> Set[str]:
    """Return built-in stop words for the language. Empty set if unavailable."""
    if language in _STOPWORDS_CACHE:
        return _STOPWORDS_CACHE[language]
    words = set(_BUILTIN_STOPWORDS.get(language, ()))
    _STOPWORDS_CACHE[language] = words
    return words


def create_tokenizer(
    language: str = "english",
    remove_stop_words: bool = False,
) -> Callable[[str], List[str]]:
    """
    Create a tokenizer with the given language and stop-word settings.

    :param language: Language code for stemmer and stop words (e.g. "english", "spanish").
    :param remove_stop_words: If True, filter out stop words. Default False (backwards compatible).
    :return: A tokenize function.
    """
    stemmer = snowballstemmer.stemmer(language)
    stop_words: Set[str] = _get_stop_words(language) if remove_stop_words else set()

    def tokenize(text: str) -> List[str]:
        if not text:
            return []

        normalized = unicodedata.normalize("NFKC", text).lower()
        raw_tokens = [
            t for t in TOKEN_SPLIT_PATTERN.split(normalized) if t
        ]
        if not raw_tokens:
            return []

        stemmed = stemmer.stemWords(raw_tokens)
        if stop_words:
            return [t for t in stemmed if t and t not in stop_words]
        return [t for t in stemmed if t]

    return tokenize


def default_tokenize_text(
    text: str,
    language: str = "english",
    remove_stop_words: bool = False,
) -> List[str]:
    """
    Normalizes, tokenizes, stems, and optionally removes stop words.

    :param text: Input text.
    :param language: Language code. Default "english".
    :param remove_stop_words: If True, filter stop words. Default False (backwards compatible).
    :return: List of tokens.
    """
    return create_tokenizer(language=language, remove_stop_words=remove_stop_words)(
        text
    )
