import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class MultivariateAnalysis(ABC):
    @abstractmethod
    def multivariate_analysis(self, df: pd.DataFrame):
        pass

class DataAggregation(MultivariateAnalysis):
    def multivariate_analysis(self, df: pd.DataFrame):
        self.df = df

    def top_billing_by_age_medical_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Age Group', 'Medical Condition', 'Gender']).agg({
            'Billing Amount': 'mean'
        }).reset_index().sort_values(ascending=False, by='Billing Amount').head(10)

    def median_billing_by_admission_insurance(self) -> pd.DataFrame:
        return self.df.groupby(['Admission Type', 'Insurance Provider']).agg({
            'Billing Amount': 'median'
        }).reset_index().head(10)

    def patient_count_by_blood_age_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Blood Type', 'Age Group', 'Gender']).size().reset_index(name='Patient Count').sort_values(by='Patient Count', ascending=False).head(10)

    def mean_billing_room_by_age_admission(self) -> pd.DataFrame:
        return self.df.groupby(['Age Group', 'Admission Type']).agg({
            'Billing Amount': 'mean',
            'Room Number': 'mean'
        }).reset_index()

    def average_length_of_stay_by_age_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Age Group', 'Gender']).agg({
            'Length of Stay': 'mean'
        }).reset_index().sort_values(ascending=False, by='Length of Stay').head(10)

    def top_medications_by_medical_condition(self) -> pd.DataFrame:
        return self.df.groupby(['Medical Condition', 'Medication']).size().reset_index(name='Count').sort_values(by='Count', ascending=False).head(10)

    def patient_count_by_medical_age_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Medical Condition', 'Age Group', 'Gender']).size().reset_index(name='Patient Count').sort_values(by='Patient Count', ascending=False).head(10)

    def avg_billing_by_admission_medical_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Admission Type', 'Medical Condition', 'Gender']).agg({
            'Billing Amount': 'mean'
        }).reset_index().sort_values(ascending=False, by='Billing Amount').head(10)

    def median_length_of_stay_by_age_insurance(self) -> pd.DataFrame:
        return self.df.groupby(['Age Group', 'Insurance Provider']).agg({
            'Length of Stay': 'median'
        }).reset_index().sort_values(ascending=False, by='Length of Stay').head(10)

    def avg_billing_by_blood_age_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Blood Type', 'Age Group', 'Gender']).agg({
            'Billing Amount': 'mean'
        }).reset_index().sort_values(ascending=False, by='Billing Amount').head(10)

    def patient_count_by_admission_age_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Admission Type', 'Age Group', 'Gender']).size().reset_index(name='Patient Count').sort_values(by='Patient Count', ascending=False).head(10)

    def avg_billing_by_medication_age_gender(self) -> pd.DataFrame:
        return self.df.groupby(['Medication', 'Age Group', 'Gender']).agg({
            'Billing Amount': 'mean'
        }).reset_index().sort_values(ascending=False, by='Billing Amount').head(10)

    def avg_length_of_stay_by_admission_age_gender(self) -> pd.DataFrame:
        result = self.df.groupby(['Admission Type', 'Age Group', 'Gender']).agg({
            'Length of Stay': 'mean'
        }).reset_index().sort_values(ascending=False, by='Length of Stay').head(10)
        result['Length of Stay'] = result['Length of Stay'].round(0)
        return result
    
class CorrelationHeatmap(MultivariateAnalysis):
    def multivariate_analysis(self, df: pd.DataFrame):
        df = df.select_dtypes(include=['number'])
        plt.figure(figsize=(10, 7))
        sns.heatmap(data=df.corr(), cmap='crest', annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

class MultivariateAnalysisFactory:
    def __init__(self, strategy: MultivariateAnalysis):
        self.strategy = strategy
    
    def analyze(self, df: pd.DataFrame):
        return self.strategy.multivariate_analysis(df)

if __name__ == "__main__":
    pass