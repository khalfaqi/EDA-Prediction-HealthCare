from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class BivariateAnalysis(ABC):
    @abstractmethod
    def bivariate_analysis(self, df: pd.DataFrame, feature1 : str, feature2: str):
        pass

class ScatterplotAnalysis(BivariateAnalysis):
    def bivariate_analysis(self, df: pd.DataFrame, feature1 : str, feature2: str):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"Scatter Plot of {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class LineplotAnalysis(BivariateAnalysis):
    def bivariate_analysis(self, df: pd.DataFrame, feature1 : str, feature2: str):
        plt.figure(figsize=(10,5))
        sns.lineplot(x=feature1, y=feature2, data=df)
        plt.title(f"Line Plot of {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class BoxplotAnalysis(BivariateAnalysis):
    def bivariate_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"Box Plot of {feature2} by {feature1}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

class BarplotAnalysis(BivariateAnalysis):
    def bivariate_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature1, y=feature2, data=df)
        plt.title(f"Bar Plot of {feature2} by {feature1}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

class HistogramAnalysis(BivariateAnalysis):
    def bivariate_analysis(self, df: pd.DataFrame, feature1: str, feature2: str, bins: int = 30):
        plt.figure(figsize=(8, 5))
        plt.hist2d(x=df[feature1], y=df[feature2], bins=bins, cmap='BuPu')
        plt.title(f"Hist 2d Plot of {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.colorbar(label='Counts')
        plt.show()

class BivariateAnalysisFactory:
    def __init__(self, strategy : BivariateAnalysis):
        self.strategy = strategy

    def analyze(self, df : pd.DataFrame, feature1 : str, feature2: str):
        return self.strategy.bivariate_analysis(df, feature1, feature2)
    
if __name__ == "__main__":
    pass