from simplebayes.tokenization import default_tokenize_text


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
