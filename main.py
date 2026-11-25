
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predicts whether a customer will subscribe to a term deposit.",
    version="1.0.0"
)

# Load the trained pipeline once
model = joblib.load(
    Path(__file__).parent / "model.joblib"
)


class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.get("/")
def home():
    return {
        "message": "Bank Marketing Prediction API is running"
    }


@app.post("/predict")
def predict(customer: Customer):

    input_data = pd.DataFrame([
        customer.model_dump()
    ])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": str(prediction)
    }
