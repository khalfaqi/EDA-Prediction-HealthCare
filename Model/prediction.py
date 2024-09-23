import pandas as pd
import joblib
from abc import ABC, abstractmethod

class Pred(ABC):
    @abstractmethod
    def pred(self, df : pd.DataFrame):
        pass

class ModelPredictor(Pred):
    def __init__(self, model_filename=r'D:\Machine Learning Project\Health Care\model.pkl'):
        self.model_filename = model_filename

    def pred(self, df: pd.DataFrame):
        model = joblib.load(self.model_filename)
        y_pred = model.predict(df)
        print("Predictions made.")
        return y_pred

class ModelPredictorFactory:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor

    def run(self, df: pd.DataFrame):
        return self.predictor.pred(df)

if __name__ == "__main__":
    pass
