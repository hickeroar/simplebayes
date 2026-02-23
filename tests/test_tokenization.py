from simplebayes.tokenization import (
    _get_stop_words,
    create_tokenizer,
    default_tokenize_text,
)


def test_default_tokenize_text_empty():
    assert default_tokenize_text("") == []


def test_default_tokenize_text_only_separators():
    assert default_tokenize_text("!!! ---") == []


def test_default_tokenize_text_normalizes_and_splits():
    tokens = default_tokenize_text("Hello, WORLD!! 123")
    assert tokens == ["hello", "world", "123"]


def test_default_tokenize_text_stems_words():
    tokens = default_tokenize_text("running runner runs")
    assert tokens == ["run", "runner", "run"]


def test_default_tokenize_text_nfkc_normalization():
    tokens = default_tokenize_text("Ｆｏｏ Bar")
    assert tokens == ["foo", "bar"]


def test_default_tokenize_text_handles_combining_marks():
    tokens = default_tokenize_text("Cafe\u0301")
    assert tokens == ["café"]


def test_default_tokenize_text_handles_zero_width_spacing():
    tokens = default_tokenize_text("alpha\u200bbeta")
    assert tokens == ["alpha", "beta"]


def test_default_tokenize_text_retains_stop_words_by_default():
    """Default remove_stop_words=False keeps stop words (backwards compatible)."""
    tokens = default_tokenize_text("the cat is in the hat")
    assert "the" in tokens
    assert "is" in tokens
    assert "in" in tokens


def test_default_tokenize_text_with_remove_stop_words_true_filters_stop_words():
    tokens = default_tokenize_text(
        "the cat is in the hat", remove_stop_words=True
    )
    assert "the" not in tokens
    assert "is" not in tokens
    assert "in" not in tokens
    assert "cat" in tokens or "hat" in tokens  # content words retained


def test_tokenizer_remove_stop_words_false_retains_all():
    tokenize = create_tokenizer(language="english", remove_stop_words=False)
    tokens = tokenize("the cat is in the hat")
    assert len(tokens) > 2  # stop words retained


def test_create_tokenizer_language_spanish():
    tokenize = create_tokenizer(language="spanish", remove_stop_words=True)
    tokens = tokenize("el gato está en la casa")
    assert "el" not in tokens
    assert "la" not in tokens
    assert len(tokens) >= 2  # content words (gato/casa) retained, possibly stemmed


def test_create_tokenizer_language_french_has_stopwords():
    """All Snowball languages have built-in stopwords."""
    tokenize = create_tokenizer(language="french", remove_stop_words=True)
    tokens = tokenize("le chat est dans la maison")
    assert "le" not in tokens
    assert "la" not in tokens
    assert "est" not in tokens


def test_create_tokenizer_language_yiddish_has_stopwords():
    """Yiddish has stopwords from Wiktionary/Wortschatz Leipzig frequency list."""
    tokenize = create_tokenizer(language="yiddish", remove_stop_words=True)
    tokens = tokenize("די וועלט איז גרויס")  # "The world is big"
    assert "די" not in tokens
    assert "איז" not in tokens
    assert len(tokens) >= 2  # content words (world, big) retained


def test_get_stop_words_unknown_language_returns_empty():
    words = _get_stop_words("nonexistentlangxyz123")
    assert words == set()


def test_get_stop_words_caches_result():
    """Second call for same language returns cached result."""
    first = _get_stop_words("english")
    second = _get_stop_words("english")
    assert first is second
    assert "the" in first
