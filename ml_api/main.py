from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

models = {}

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


# Load the saved model when the API starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    models["predictor"] = joblib.load(MODEL_PATH)
    yield
    models.clear()


app = FastAPI(
    title="Student Prediction API",
    lifespan=lifespan,
)


# Inputs sent by the user.
class StudentFeatures(BaseModel):
    study_hours: float = Field(ge=0, le=10)
    prev_score: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Prediction API is running"}


@app.post("/predict")
def predict(features: StudentFeatures):
    model = models["predictor"]

    inputs = np.array([[
        features.study_hours,
        features.prev_score,
    ]])

    prediction = int(model.predict(inputs)[0])

    probabilities = model.predict_proba(inputs)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    return {
        "prediction": prediction,
        "result_label": "Pass" if prediction == 1 else "Fail",
        "confidence": round(confidence, 3),
    }