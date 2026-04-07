from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("mlfp/index.html", encoding="utf-8") as f:
        return f.read()
    
class InputData(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    hours_used: float    

@app.post("/predict")
def predict(data: InputData):
    values = np.array([
        data.temperature,
        data.vibration,
        data.pressure,
        data.hours_used
    ]).reshape(1, -1)

    prediction = model.predict(values)[0]

    return {"prediction": int(prediction)}