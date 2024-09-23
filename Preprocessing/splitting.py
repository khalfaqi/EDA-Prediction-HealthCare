import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class Splitting(ABC):
    @abstractmethod
    def traintestsplit(self, df: pd.DataFrame):
        pass

class TrainTestSplit(Splitting):
    def traintestsplit(self, df : pd.DataFrame):
        X = df.drop(['Test Results'], axis=1)
        y = df['Test Results']
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test
    
class TrainTestSplitFactory:
    def __init__(self, strategy : Splitting):
        self.strategy = strategy

    def splitting(self, df : pd.DataFrame):
        return self.strategy.traintestsplit(df)
    
if __name__ == "__main__":
    pass