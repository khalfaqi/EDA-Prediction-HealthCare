import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import joblib

class Model(ABC):
    @abstractmethod
    def build_model(self, X_train, y_train):
        pass

class RFC(Model):
    def build_model(self, X_train, y_train, model_filename='model.pkl'):
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        # Save the model to a file
        joblib.dump(rfc, model_filename)
        print(f"Model saved to {model_filename}")
        return rfc
        
class ModelFactory:
    def __init__(self, strategy : Model):
        self.strategy = strategy

    def execute(self, X_train, y_train):
        return self.strategy.build_model(X_train, y_train)
    
if __name__ == "__main__":
    pass

