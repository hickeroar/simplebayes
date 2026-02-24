from simplebayes import cli


def test_parse_args_defaults(monkeypatch):
    monkeypatch.delenv("SIMPLEBAYES_HOST", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_PORT", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_LANGUAGE", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_REMOVE_STOP_WORDS", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_VERBOSE", raising=False)

    args = cli.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.auth_token == ""
    assert args.language == "english"
    assert args.remove_stop_words is False
    assert args.verbose is False


def test_parse_args_uses_env(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_HOST", "127.0.0.1")
    monkeypatch.setenv("SIMPLEBAYES_PORT", "9000")
    monkeypatch.setenv("SIMPLEBAYES_AUTH_TOKEN", "env-token")

    args = cli.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 9000
    assert args.auth_token == "env-token"


def test_parse_args_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_HOST", "127.0.0.1")
    monkeypatch.setenv("SIMPLEBAYES_PORT", "9000")
    monkeypatch.setenv("SIMPLEBAYES_AUTH_TOKEN", "env-token")

    args = cli.parse_args(["--host", "localhost", "--port", "8123", "--auth-token", "cli-token"])
    assert args.host == "localhost"
    assert args.port == 8123
    assert args.auth_token == "cli-token"


def test_parse_args_language_default(monkeypatch):
    monkeypatch.delenv("SIMPLEBAYES_LANGUAGE", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_REMOVE_STOP_WORDS", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_VERBOSE", raising=False)
    args = cli.parse_args([])
    assert args.language == "english"


def test_parse_args_language_cli():
    args = cli.parse_args(["--language", "spanish"])
    assert args.language == "spanish"


def test_parse_args_language_env(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_LANGUAGE", "spanish")
    args = cli.parse_args([])
    assert args.language == "spanish"


def test_parse_args_remove_stop_words_default(monkeypatch):
    monkeypatch.delenv("SIMPLEBAYES_REMOVE_STOP_WORDS", raising=False)
    args = cli.parse_args([])
    assert args.remove_stop_words is False


def test_parse_args_remove_stop_words_flag():
    args = cli.parse_args(["--remove-stop-words"])
    assert args.remove_stop_words is True


def test_parse_args_remove_stop_words_env(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_REMOVE_STOP_WORDS", "1")
    args = cli.parse_args([])
    assert args.remove_stop_words is True


def test_parse_args_remove_stop_words_env_yes(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_REMOVE_STOP_WORDS", "yes")
    args = cli.parse_args([])
    assert args.remove_stop_words is True


def test_parse_args_remove_stop_words_env_false(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_REMOVE_STOP_WORDS", "0")
    args = cli.parse_args([])
    assert args.remove_stop_words is False


def test_parse_args_verbose_default(monkeypatch):
    monkeypatch.delenv("SIMPLEBAYES_VERBOSE", raising=False)
    args = cli.parse_args([])
    assert args.verbose is False


def test_parse_args_verbose_flag():
    args = cli.parse_args(["--verbose"])
    assert args.verbose is True


def test_parse_args_verbose_env(monkeypatch):
    monkeypatch.setenv("SIMPLEBAYES_VERBOSE", "true")
    args = cli.parse_args([])
    assert args.verbose is True


def test_run_invokes_uvicorn(monkeypatch):
    captured = {}

    def fake_create_app(
        auth_token: str = "",
        language: str = "english",
        remove_stop_words: bool = False,
        verbose: bool = False,
    ):
        captured["auth_token"] = auth_token
        captured["language"] = language
        captured["remove_stop_words"] = remove_stop_words
        captured["verbose"] = verbose
        return "app-object"

    def fake_uvicorn_run(app, host, port):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_uvicorn_run)

    cli.run(["--host", "localhost", "--port", "8181", "--auth-token", "top-secret"])

    assert captured["auth_token"] == "top-secret"
    assert captured["language"] == "english"
    assert captured["remove_stop_words"] is False
    assert captured["verbose"] is False
    assert captured["app"] == "app-object"
    assert captured["host"] == "localhost"
    assert captured["port"] == 8181


def test_run_passes_language_remove_stop_words_verbose(monkeypatch):
    captured = {}

    def fake_create_app(
        auth_token: str = "",
        language: str = "english",
        remove_stop_words: bool = False,
        verbose: bool = False,
    ):
        captured["auth_token"] = auth_token
        captured["language"] = language
        captured["remove_stop_words"] = remove_stop_words
        captured["verbose"] = verbose
        return "app-object"

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)

    cli.run(
        [
            "--auth-token",
            "x",
            "--language",
            "spanish",
            "--remove-stop-words",
            "--verbose",
        ]
    )

    assert captured["language"] == "spanish"
    assert captured["remove_stop_words"] is True
    assert captured["verbose"] is True
