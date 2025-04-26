from fastapi import FastAPI
from pydantic import BaseModel
from server.classifier.model import IrisModel

app = FastAPI()
model = IrisModel("server/models/iris_model.pkl")

class Features(BaseModel):
    data: list

@app.post("/predict")
def predict(features: Features):
    prediction = model.predict(features.data)
    return {"prediction": prediction}
