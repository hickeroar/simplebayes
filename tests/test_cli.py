from simplebayes import cli


def test_parse_args_defaults(monkeypatch):
    monkeypatch.delenv("SIMPLEBAYES_HOST", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_PORT", raising=False)
    monkeypatch.delenv("SIMPLEBAYES_AUTH_TOKEN", raising=False)

    args = cli.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.auth_token == ""


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


def test_run_invokes_uvicorn(monkeypatch):
    captured = {}

    def fake_create_app(auth_token: str = ""):
        captured["auth_token"] = auth_token
        return "app-object"

    def fake_uvicorn_run(app, host, port):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_uvicorn_run)

    cli.run(["--host", "localhost", "--port", "8181", "--auth-token", "top-secret"])

    assert captured["auth_token"] == "top-secret"
    assert captured["app"] == "app-object"
    assert captured["host"] == "localhost"
    assert captured["port"] == 8181
