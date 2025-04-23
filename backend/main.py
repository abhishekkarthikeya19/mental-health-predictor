# main.py
from fastapi import FastAPI

app = FastAPI()

# Define the root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Predictor API!"}

# Add other routes as needed

