from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# CORS (so frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../model/model.joblib")
model = joblib.load(model_path)

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    text = data["text_input"]
    
    # Convert input into numeric feature (length of input text)
    df = pd.DataFrame([{"text_input": len(text)}])
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}
