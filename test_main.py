from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

VALID_PAYLOAD = {
    "age": 35,
    "annual_income": 62000,
    "loan_amount": 15000,
    "credit_score": 700,
    "employment_years": 6,
    "education": "Bachelor",
    "home_ownership": "Rent",
    "loan_purpose": "Car"
}

def test_root_and_health():
    assert client.get("/").status_code == 200
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_prediction_schema_when_model_available():
    if main.model is None:
        # The API should report a clear service-unavailable response until a model is trained.
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
    else:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in [0, 1]
        assert body["label"] in ["Default risk", "No default risk"]
        assert "probabilities" in body
        assert "model_name" in body

def test_invalid_payload_returns_422():
    bad = dict(VALID_PAYLOAD)
    bad["age"] = 4
    assert client.post("/predict", json=bad).status_code == 422

def test_extra_fields_are_rejected():
    bad = dict(VALID_PAYLOAD, unexpected_field="not allowed")
    assert client.post("/predict", json=bad).status_code == 422
