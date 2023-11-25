import unittest
from fastapi.testclient import TestClient
from main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.API_KEY = "your-api-key"

    def test_predict_with_valid_data(self):
        data = [1.0, 2.0, 3.0, 4.0]
        headers = {"X-API-Key": self.API_KEY}

        response = self.client.post("/predict", headers=headers, json=data)
        result = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("result", result)
        self.assertEqual(result["result"], [2.0, 4.0, 6.0, 8.0])

    def test_predict_with_invalid_api_key(self):
        data = [1.0, 2.0, 3.0, 4.0]
        headers = {"X-API-Key": "invalid-api-key"}

        response = self.client.post("/predict", headers=headers, json=data)

        self.assertEqual(response.status_code, 403)

    def test_predict_with_invalid_data(self):
        data = "invalid-data"  # Data should be a list of floats
        headers = {"X-API-Key": self.API_KEY}

        response = self.client.post("/predict", headers=headers, json=data)

        self.assertEqual(response.status_code, 422)  # 422 Unprocessable Entity

if __name__ == "__main__":
    unittest.main()
