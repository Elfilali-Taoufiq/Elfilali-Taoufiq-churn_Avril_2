import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

class DummyModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


@pytest.fixture
def client(monkeypatch):
    # Mock du modèle
    from app import model
    monkeypatch.setattr("app.model", DummyModel())

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_home(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_endpoint(client):
    payload = {
        "features": [30, 1, 5, 10]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.get_json()

    assert "prediction" in data
    assert isinstance(data["prediction"], list)


def test_predict_invalid_input(client):
    payload = {
        "wrong_key": [1, 2, 3]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    data = response.get_json()

    assert "error" in data


def test_predict_ui(client):
    response = client.post("/predict-ui", data={
        "Age": "30",
        "Account_Manager": "1",
        "Years": "5",
        "Num_Sites": "10"
    })

    assert response.status_code == 200
    assert b"Prediction" in response.data or b"Churn" in response.data