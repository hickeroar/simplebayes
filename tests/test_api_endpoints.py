from fastapi.testclient import TestClient

from simplebayes.api.app import create_app


def test_health_and_ready_endpoints():
    client = TestClient(create_app())
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


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
