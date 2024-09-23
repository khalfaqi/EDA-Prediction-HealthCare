import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler

class Scaling(ABC):
    @abstractmethod
    def feature_scale(self, df : pd.DataFrame):
        pass

class FeatScale:
    def feature_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # Columns that need scaling
        columns_to_scale = ['Age', 'Billing Amount', 'Room Number']
        
        # Select columns to scale
        df_to_scale = df[columns_to_scale]

        ss = StandardScaler()
        scaled_values = ss.fit_transform(df_to_scale)
        
        # Create a DataFrame with scaled values
        scaled_df = pd.DataFrame(scaled_values, columns=columns_to_scale, index=df.index)
        
        # Combine the scaled DataFrame with the columns that do not need scaling
        columns_not_to_scale = [col for col in df.columns if col not in columns_to_scale]
        df_not_to_scale = df[columns_not_to_scale]
        
        # Concatenate scaled and non-scaled DataFrames
        df_combined = pd.concat([scaled_df, df_not_to_scale], axis=1)
        
        return df_combined

class ScalingFactory:
    def __init__(self, strategy : Scaling):
        self.strategy = strategy

    def feature(self, df : pd.DataFrame):
        return self.strategy.feature_scale(df)
    
if __name__ == "__main__":
    pass