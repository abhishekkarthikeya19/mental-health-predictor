from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("app/model/mental_health_model.pkl")

@app.get("/")
def read_root():
    return {"message": "API running"}

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    text = data["text_input"]
    # Pass just the text input to the model's predict method
    prediction = model.predict([text])[0]
    return {"prediction": int(prediction)}
