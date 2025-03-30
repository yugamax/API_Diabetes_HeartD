import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import uvicorn

tf.config.optimizer.set_jit(True) 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scalers
def load_models():
    global diabetes_model, heart_disease_model, diabetes_scaler, heart_scaler
    diabetes_model = tf.keras.models.load_model("models/diabetes_model.keras")
    heart_disease_model = tf.keras.models.load_model("models/heart_disease_model.keras")
    diabetes_scaler = joblib.load("models/db_scaler.joblib")
    heart_scaler = joblib.load("models/hd_scaler.joblib")

load_models()

class DiabetesInput(BaseModel):
    glucose: float
    BP: float
    insulin: float
    BMI: float
    age: float

class HeartDiseaseInput(BaseModel):
    age: int
    gender: int
    cp: int
    trestbps: int
    chol: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int

def process_diabetes(input_data: DiabetesInput):
    inp_arr = np.array([list(input_data.dict().values())])
    return diabetes_scaler.transform(inp_arr)

def process_heart_disease(input_data: HeartDiseaseInput):
    inp_arr = np.array([list(input_data.dict().values())])
    return heart_scaler.transform(inp_arr)

@app.post("/predict/diabetes")
async def predict_diabetes(input_data: DiabetesInput):
    try:
        input_scaled = process_diabetes(input_data)
        result = diabetes_model.predict(input_scaled)[0][0]
        diagnosis = "You have diabetes" if result > 0.5 else "You don't have diabetes"
        return {"result": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/heart_disease")
async def predict_heart_disease(input_data: HeartDiseaseInput):
    try:
        input_scaled = process_heart_disease(input_data)
        result = heart_disease_model.predict(input_scaled)[0][0]
        diagnosis = "You have Heart disease" if result > 0.5 else "You don't have Heart disease"
        return {"result": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)