from fastapi.testclient import TestClient
from src.main import app
from pathlib import Path
from dotenv import dotenv_values

env_path = Path(".") / ".env"

config = dotenv_values(env_path)


def test_predict_with_valid_data():
    with TestClient(app) as client:

        # Mock data for the request
        data = {"data": [1.0, 2.0, 3.0]}
        headers = {
            "accept": "application/json",
            "X-API-Key": config["API_KEY"],  # Replace with your actual API key
            "Content-Type": "application/json",
        }

        response = client.post("/predict", headers=headers, json=data)

        assert response.status_code == 200
        assert "results" in response.json()
        assert response.json()["results"]["result"] == [2.0, 4.0, 6.0]
        assert response.json()["error"] == False


def test_predict_with_invalid_api_key():
    client = TestClient(app)
    data = [1.0, 2.0, 3.0, 4.0]
    headers = {"X-API-Key": "invalid-api-key",
               "Content-Type": "application/json",
               "Accept": "application/json"}

    response = client.post("/predict", headers=headers, json=data)

    assert response.status_code == 403


def test_predict_with_invalid_data():
    client = TestClient(app)
    API_KEY = config["API_KEY"]
    data = "invalid-data"  # Data should be a list of floats
    headers = {"X-API-Key": API_KEY,
               "Content-Type": "application/json",
               "Accept": "application/json"}

    response = client.post("/predict", headers=headers, json=data)

    assert response.status_code == 422  # 422 Unprocessable Entity

