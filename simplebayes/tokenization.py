import re
import unicodedata
from typing import List

import snowballstemmer

TOKEN_SPLIT_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
STEMMER = snowballstemmer.stemmer("english")


def default_tokenize_text(text: str) -> List[str]:
    """Normalizes, tokenizes, and stems input text."""
    if not text:
        return []

    normalized = unicodedata.normalize("NFKC", text).lower()
    raw_tokens = [token for token in TOKEN_SPLIT_PATTERN.split(normalized) if token]
    if not raw_tokens:
        return []

    stemmed_tokens = STEMMER.stemWords(raw_tokens)
    return [token for token in stemmed_tokens if token]
