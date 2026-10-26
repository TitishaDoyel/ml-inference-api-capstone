# Task 06 — Real-Time ML Inference REST API & Capstone

A FastAPI service that serves a trained scikit-learn credit-default classification pipeline. Includes request validation, prediction probabilities, automated tests, Docker packaging, and API documentation.

> **Demo-data notice:** `train_model.py` creates a simulated dataset to make the project runnable. This is a learning/demo implementation, not a credit decision system. Replace the simulated data with an appropriately licensed real dataset and validate the model before real-world use.

## Project files

- `main.py` — FastAPI routes: `/`, `/health`, and `/predict`
- `train_model.py` — trains and saves the champion scikit-learn pipeline
- `test_main.py` — API schema, validation, and response tests
- `requirements.txt` — pinned Python dependencies
- `Dockerfile` — container build and startup instructions

## Run locally

Use Python 3.11.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

## Example request

POST to `/predict` with JSON:

```json
{
  "age": 35,
  "annual_income": 62000,
  "loan_amount": 15000,
  "credit_score": 700,
  "employment_years": 6,
  "education": "Bachelor",
  "home_ownership": "Rent",
  "loan_purpose": "Car"
}
```

Example response structure:

```json
{
  "prediction": 0,
  "label": "No default risk",
  "probabilities": {
    "0": 0.72,
    "1": 0.28
  },
  "model_name": "LogisticRegression"
}
```

The numeric probabilities and selected model vary with the training run; the example above is illustrative.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service welcome message |
| GET | `/health` | Health check and model-loaded status |
| POST | `/predict` | Validates input JSON and returns class + probabilities |
| GET | `/docs` | Interactive OpenAPI / Swagger UI |

Invalid input returns HTTP `422`. If the model file is missing, `/predict` returns HTTP `503` with setup guidance.

## Run tests

```bash
python train_model.py
pytest -q
```

## Build and run with Docker

Docker must be installed and running.

```bash
docker build -t task6-ml-api .
docker run --rm -p 8000:8000 task6-ml-api
```

Then visit `http://localhost:8000/docs`.

## Architecture

1. Client sends JSON to `POST /predict`.
2. FastAPI/Pydantic validates the fields and value ranges.
3. The saved scikit-learn pipeline imputes missing values, scales numeric features, encodes categorical features, and predicts.
4. The API returns the predicted class, readable label, class probabilities, and model name.
5. Docker packages the Python dependencies and service into a reproducible container.

## Production considerations

- Replace the demo data with a documented, appropriately licensed dataset.
- Track model/data versions and monitor drift.
- Add authentication, rate limiting, structured logs, and observability before public deployment.
- Avoid sending personal or financial data to a demo API.
- A probability is a model estimate, not a guarantee or a final lending decision.
