from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "ML API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    values = np.array([
        data["temperature"],
        data["vibration"],
        data["pressure"],
        data["hours_used"]
    ]).reshape(1, -1)

    prediction = model.predict(values)[0]

    return {"prediction": int(prediction)}