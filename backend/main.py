# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_path = "app/model/mental_health_model.pkl"
model = joblib.load(model_path)

# Define the root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Predictor API!"}

# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    text = data["text_input"]
    df = pd.DataFrame([{"text_input": text}])
    prediction = model.predict(df["text_input"])[0]
    return {"prediction": int(prediction)}

