from fastapi import FastAPI
import joblib
import os
import torch

# Make sure PyTorch is installed in your environment before running this script

# Initialize FastAPI
app = FastAPI()

# Define the path to the saved model
model_path = os.path.join("model", "model.joblib")

# Try to load the model
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None  # Avoid crash if model is missing
    print(f"⚠️ Could not load model: {e}")

# Load the PyTorch model
model = torch.load('model.pth')
model.eval()

# Define a simple test route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Predictor API!"}
