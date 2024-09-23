import pandas as pd
from abc import ABC, abstractmethod

class DataInspection(ABC):
    @abstractmethod
    def data_inspection(self, df: pd.DataFrame):
        pass

class DataInfo(DataInspection):
    def data_inspection(self, df: pd.DataFrame) -> None:
        df.info()  

class SummaryStatistics(DataInspection):
    def data_inspection(self, df: pd.DataFrame) -> None:
        print(df.describe())  

class DescriptiveStatistics(DataInspection):
    def data_inspection(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.select_dtypes(include=['int64', 'float64'])
        desc = pd.DataFrame()
        desc['Sample Count'] = df.count()
        desc['Missing Values'] = df.isnull().sum()
        desc['Number of Unique'] = df.nunique()
        desc['Unique (%)'] = df.nunique() / df.shape[0] * 100
        desc = desc.join(df.describe().T.drop(columns='count'))
        return desc 

class DataInspectionFactory:
    def __init__(self, strategy: DataInspection):
        self.strategy = strategy

    def data_set_strategy(self, strategy: DataInspection):
        self.strategy = strategy

    def data_inspection_execute(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.data_inspection(df)  


if __name__ == "__main__":
    pass
