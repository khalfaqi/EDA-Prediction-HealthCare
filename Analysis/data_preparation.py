from abc import ABC, abstractmethod
import pandas as pd

class DataFixer(ABC):
    @abstractmethod
    def data_fixing(self, df: pd.DataFrame):
        pass

class Preparation(DataFixer):
    def data_fixing(self, df: pd.DataFrame):
        def categorize_age(age):
            if age < 20:
                return 'Children/Teenagers'
            elif 20 <= age < 40:
                return 'Young Adults'
            elif 40 <= age < 60:
                return 'Middle-aged Adults'
            else:
                return 'Seniors'

        
        df['Age Group'] = df['Age'].apply(categorize_age)
        
        # Length of Stay calculation
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
        df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
        
        # Billing Amount categorization
        def categorize_billing(amount):
            if amount < 1000:
                return 'Low'
            elif 1000 <= amount < 5000:
                return 'Medium'
            elif 5000 <= amount < 10000:
                return 'High'
            else:
                return 'Very High'
        
        df['Billing Category'] = df['Billing Amount'].apply(categorize_billing)
        
        # Standardizing Gender format
        df['Gender'] = df['Gender'].str.capitalize()
        
        # Standardizing Insurance Provider format
        df['Insurance Provider'] = df['Insurance Provider'].str.strip().str.title()
        
        # Standardizing Admission Type format
        df['Admission Type'] = df['Admission Type'].str.title()
        
        return df

class PreparationFactory:
    def __init__(self, strategy : DataFixer):
        self.strategy = strategy

    def prepare(self, df : pd.DataFrame):
        return self.strategy.data_fixing(df)
    
if __name__ == "__main__":
    pass