import pytest
from fastapi.testclient import TestClient
from src.app import app
import src.models.predict as predict

client = TestClient(app)


@pytest.fixture(autouse=True)
def patch_predict(monkeypatch):
    # Replace the real predict function with a dummy one
    monkeypatch.setattr(predict, "predict", lambda data: "Gentoo")
    yield


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Penguin ML is running"}


def test_predict_endpoint():
    payload = {
        "bill_length_mm": 50.0,
        "bill_depth_mm": 15.0,
        "flipper_length_mm": 220.0,
        "body_mass_g": 4500.0
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json() == {"predicted_species": "Gentoo"}
