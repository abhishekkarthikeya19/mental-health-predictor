# main.py
# Apply patch for anyio to avoid uvloop on Windows
import sys
if sys.platform == 'win32':
    try:
        # Try file-based patching first
        from patch_anyio import patch_anyio
        patch_anyio()
    except ImportError:
        print("Warning: patch_anyio module not found. Trying runtime patching.")
    
    try:
        # Try runtime patching as a fallback
        from runtime_patch import patch_anyio_at_runtime
        patch_anyio_at_runtime()
    except ImportError:
        print("Warning: runtime_patch module not found. uvloop errors may occur on Windows.")

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
model_path = "../app/model/mental_health_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    # Try alternative path
    model_path = "app/model/mental_health_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        # Try one more path
        model_path = "./app/model/mental_health_model.pkl"
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
    # Pass just the text input to the model's predict method
    prediction = model.predict([text])[0]
    return {"prediction": int(prediction)}

