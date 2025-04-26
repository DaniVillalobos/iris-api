import joblib
import numpy as np

class IrisModel:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, features: list):
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        return int(prediction[0])
