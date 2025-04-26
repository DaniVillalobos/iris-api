from fastapi import FastAPI
import requests
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/classify")
def classify(data: InputData):
    response = requests.post("http://localhost:8001/predict", json={"data": data.features})
    prediction = response.json()
    return {"class": prediction["prediction"]}
