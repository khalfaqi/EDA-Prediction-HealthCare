import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

class Encoding(ABC):
    @abstractmethod
    def encoding(self, df : pd.DataFrame):
        pass

class OrdinalEncode(Encoding):
    def encoding(self, df: pd.DataFrame):
        df_to_encode = df[['Gender','Admission Type', 'Medical Condition', 'Insurance Provider', 'Medication']]
        oe = OrdinalEncoder()
        df_encoded = pd.DataFrame(oe.fit_transform(df_to_encode), 
                                  columns=df_to_encode.columns)
        df_final = pd.concat([df.drop(columns=['Gender','Admission Type', 'Medical Condition', 'Insurance Provider', 'Medication']), df_encoded], axis=1)
        return df_final


class LabelEncode(Encoding):
    def encoding(self, df: pd.DataFrame):
        df_encode = df.copy()
        df_target = df_encode['Test Results']
        le = LabelEncoder()
        df_encode['Test Results'] = le.fit_transform(df_target)
        return df_encode


class EncodingFactory:
    def __init__(self, strategy : Encoding):
        self.strategy = strategy
        
    def preprocess(self, df : pd.DataFrame):
        return self.strategy.encoding(df)
    
if __name__ == "__main__":
    pass