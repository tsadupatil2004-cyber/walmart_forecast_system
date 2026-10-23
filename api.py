from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import pydantic

app = FastAPI(title="Walmart Sales Forecasting API", version="1.0")

try:
    model = joblib.load("models/walmart_model.pkl")
except Exception as e:
    model = None

class PredictionRequest(pydantic.BaseModel):
    Store: int
    Dept: int
    IsHoliday: int
    Size: float
    Type_Encoded: int
    Temperature: float
    Fuel_Price: float
    MarkDown1: float = 0.0
    MarkDown2: float = 0.0
    MarkDown3: float = 0.0
    MarkDown4: float = 0.0
    MarkDown5: float = 0.0
    CPI: float
    Unemployment: float
    Year: int
    Month: int
    Week: int
    Day: int

@app.get("/")
def home():
    return {"status": "Online", "message": "Walmart Forecasting API is running successfully."}

@app.post("/predict")
def predict_sales(payload: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not loaded. Run train.py first.")
    
    input_df = pd.DataFrame([payload.dict()])
    prediction = model.predict(input_df)[0]
    
    return {
        "store": payload.Store,
        "department": payload.Dept,
        "predicted_weekly_sales": round(float(prediction), 2)
    }