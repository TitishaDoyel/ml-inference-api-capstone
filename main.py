from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "champion_model.joblib"

app = FastAPI(
    title="Real-Time ML Inference API",
    description="REST API for credit-default risk inference using a trained scikit-learn pipeline.",
    version="1.0.0",
)

model = None

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(gt=0, description="Annual income")
    loan_amount: float = Field(gt=0, description="Requested loan amount")
    credit_score: float = Field(ge=300, le=850, description="Credit score")
    employment_years: float = Field(ge=0, le=80, description="Years of employment")
    education: str = Field(min_length=1, max_length=50)
    home_ownership: str = Field(min_length=1, max_length=50)
    loan_purpose: str = Field(min_length=1, max_length=50)

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float]
    model_name: str

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {
        "message": "Real-Time ML Inference API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run `python train_model.py` to create champion_model.joblib, then restart the API."
        )

    row = pd.DataFrame([request.model_dump()])
    try:
        prediction = int(model.predict(row)[0])
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(row)[0]
            classes = getattr(model, "classes_", [0, 1])
            probabilities = {str(int(c)): round(float(p), 6) for c, p in zip(classes, raw)}
        else:
            probabilities = {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc

    return PredictionResponse(
        prediction=prediction,
        label="Default risk" if prediction == 1 else "No default risk",
        probabilities=probabilities,
        model_name=type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else type(model).__name__,
    )
