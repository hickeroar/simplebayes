from fastapi.testclient import TestClient
import pytest

from simplebayes.api.app import create_app


def test_health_and_ready_endpoints():
    app = create_app()
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


def test_readyz_returns_503_when_not_ready():
    app = create_app()
    app.state.readiness.mark_not_ready()
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "not ready"}


def test_lifespan_marks_not_ready_on_shutdown():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        assert app.state.readiness.is_ready is True

    assert app.state.readiness.is_ready is False


def test_train_info_score_classify_and_flush_flow():
    client = TestClient(create_app())
    headers = {"Content-Type": "text/plain"}

    train_response = client.post("/train/spam", content="buy now limited offer", headers=headers)
    assert train_response.status_code == 200
    assert train_response.json()["success"] is True

    info_response = client.get("/info")
    assert info_response.status_code == 200
    assert "spam" in info_response.json()["categories"]

    score_response = client.post("/score", content="limited offer", headers=headers)
    assert score_response.status_code == 200
    assert "spam" in score_response.json()

    classify_response = client.post("/classify", content="limited offer", headers=headers)
    assert classify_response.status_code == 200
    assert classify_response.json()["category"] == "spam"

    untrain_response = client.post("/untrain/spam", content="buy now limited offer", headers=headers)
    assert untrain_response.status_code == 200
    assert untrain_response.json()["success"] is True

    flush_response = client.post("/flush", content="", headers=headers)
    assert flush_response.status_code == 200
    assert flush_response.json() == {"success": True, "categories": {}}


def test_invalid_category_route_returns_422():
    client = TestClient(create_app())
    response = client.post(
        "/train/bad route",
        content="sample",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 422


def test_wrong_method_returns_405():
    client = TestClient(create_app())
    response = client.get("/classify")
    assert response.status_code == 405


def test_auth_required_for_non_probe_endpoints():
    client = TestClient(create_app(auth_token="secret-token"))

    unauthorized = client.get("/info")
    assert unauthorized.status_code == 401
    assert unauthorized.json() == {"error": "unauthorized"}
    assert unauthorized.headers["www-authenticate"] == 'Bearer realm="simplebayes"'

    wrong_token = client.get("/info", headers={"Authorization": "Bearer wrong-token"})
    assert wrong_token.status_code == 401
    assert wrong_token.json() == {"error": "unauthorized"}

    authorized = client.get("/info", headers={"Authorization": "Bearer secret-token"})
    assert authorized.status_code == 200


def test_probes_remain_unauthenticated_with_auth_enabled():
    client = TestClient(create_app(auth_token="secret-token"))
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_payload_too_large_returns_413():
    client = TestClient(create_app())
    too_large = "x" * (1024 * 1024 + 1)
    response = client.post(
        "/score",
        content=too_large,
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 413
    assert response.json() == {"error": "request body too large"}


def test_classify_returns_null_category_when_untrained():
    client = TestClient(create_app())
    response = client.post(
        "/classify",
        content="anything",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    assert response.json() == {"category": None, "score": 0.0}


@pytest.mark.parametrize("path", ["/train/spam", "/classify", "/score"])
def test_invalid_utf8_payload_returns_400(path):
    client = TestClient(create_app())
    response = client.post(
        path,
        content=b"\xff\xfe\xfa",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid utf-8 payload"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/train/spam"),
        ("post", "/untrain/spam"),
        ("post", "/classify"),
        ("post", "/score"),
        ("post", "/flush"),
    ],
)
def test_auth_rejected_for_each_mutating_endpoint(method, path):
    client = TestClient(create_app(auth_token="secret-token"))
    response = getattr(client, method)(
        path,
        content="body",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


def test_auth_allows_mutating_endpoints_with_valid_token():
    client = TestClient(create_app(auth_token="secret-token"))
    headers = {
        "Authorization": "Bearer secret-token",
        "Content-Type": "text/plain",
    }

    train_response = client.post("/train/spam", content="buy now limited offer", headers=headers)
    assert train_response.status_code == 200
    assert train_response.json()["success"] is True

    score_response = client.post("/score", content="limited offer", headers=headers)
    assert score_response.status_code == 200
    assert "spam" in score_response.json()

    classify_response = client.post("/classify", content="limited offer", headers=headers)
    assert classify_response.status_code == 200
    assert classify_response.json()["category"] == "spam"

    untrain_response = client.post("/untrain/spam", content="buy now limited offer", headers=headers)
    assert untrain_response.status_code == 200
    assert untrain_response.json()["success"] is True

    flush_response = client.post("/flush", content="", headers=headers)
    assert flush_response.status_code == 200
    assert flush_response.json() == {"success": True, "categories": {}}


@pytest.mark.parametrize(
    "auth_header",
    [
        "Token secret-token",
        "Bearer",
        "Bearer ",
        "Basic c2VjcmV0LXRva2Vu",
    ],
)
def test_auth_malformed_headers_are_rejected(auth_header):
    client = TestClient(create_app(auth_token="secret-token"))
    response = client.get("/info", headers={"Authorization": auth_header})
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert response.headers["www-authenticate"] == 'Bearer realm="simplebayes"'


@pytest.mark.parametrize(
    "path",
    [
        "/train/spam",
        "/untrain/spam",
        "/classify",
        "/score",
        "/flush",
    ],
)
def test_payload_exactly_one_mebibyte_is_accepted(path):
    client = TestClient(create_app())
    boundary_payload = "x" * (1024 * 1024)
    response = client.post(
        path,
        content=boundary_payload,
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/train/spam",
        "/untrain/spam",
        "/classify",
        "/score",
        "/flush",
    ],
)
def test_payload_too_large_for_each_text_endpoint(path):
    client = TestClient(create_app())
    too_large = "x" * (1024 * 1024 + 1)
    response = client.post(
        path,
        content=too_large,
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 413
    assert response.json() == {"error": "request body too large"}
