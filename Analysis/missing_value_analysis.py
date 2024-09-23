import pandas as pd
from abc import ABC, abstractmethod

class MissingValueAnalysis(ABC):
    @abstractmethod
    def missing_value_analysis(self, df: pd.DataFrame):
        pass

class MissingValuesNumber(MissingValueAnalysis):
    def missing_value_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select numerical columns (int64, float64)
        data_miss_num = df.select_dtypes(include=['number'])
        
        # Check if there are any missing values
        if data_miss_num.isna().sum().sum() > 0:
            print("There are missing values in numerical data. (Using Spline interpolation to fill missing values)")
            
            # Interpolate missing values using linear method
            interpolated_data = data_miss_num.interpolate(method='spline', order=2)
            print("Missing values have been filled.")
            
            return interpolated_data
        else:
            print("There's no missing value in numerical data.")
            return data_miss_num


class MissingValuesCategory(MissingValueAnalysis):
    def missing_value_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select non-numeric columns (categorical)
        data_miss_cat = df.select_dtypes(exclude=['number'])
        
        # Check if there are any missing values
        if data_miss_cat.isna().sum().sum() > 0:
            print("There are missing values in categorical data. (Using mode to fill missing values)")
            
            # Fill missing values with the mode (most frequent value) of each column
            filled_data = data_miss_cat.apply(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
            print("Missing values have been filled.")
            
            return filled_data
        else:
            print("There's no missing value in categorical data.")
            return data_miss_cat

class MissingValueAnalysisFactory:
    def __init__(self, strategy: MissingValueAnalysis):
        self.strategy = strategy

    def missing_value_analysis_execute(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.missing_value_analysis(df)
    
if __name__ == "__main__":
    pass