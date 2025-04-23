# main.py
from fastapi import FastAPI
import joblib
import os
import torch

# Make sure PyTorch is installed in your environment before running this script

# Initialize FastAPI
app = FastAPI()

# Define the root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Predictor API!"}

# Add other routes as needed

