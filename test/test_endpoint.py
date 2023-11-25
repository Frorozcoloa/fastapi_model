from fastapi.testclient import TestClient
from src.main import app


def test_predict_with_valid_data():
    client = TestClient(app)
    API_KEY = "your-api-key"
    data = [1.0, 2.0, 3.0, 4.0]
    headers = {"X-API-Key": API_KEY}

    response = client.post("/predict", headers=headers, json=data)
    result = response.json()

    assert response.status_code == 200
    assert "result" in result
    assert result["result"] == [2.0, 4.0, 6.0, 8.0]


def test_predict_with_invalid_api_key():
    client = TestClient(app)
    API_KEY = "your-api-key"
    data = [1.0, 2.0, 3.0, 4.0]
    headers = {"X-API-Key": "invalid-api-key"}

    response = client.post("/predict", headers=headers, json=data)

    assert response.status_code == 403


def test_predict_with_invalid_data():
    client = TestClient(app)
    API_KEY = "your-api-key"
    data = "invalid-data"  # Data should be a list of floats
    headers = {"X-API-Key": API_KEY}

    response = client.post("/predict", headers=headers, json=data)

    assert response.status_code == 422  # 422 Unprocessable Entity
