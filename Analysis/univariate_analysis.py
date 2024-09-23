from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class UnivariateAnalysis(ABC):
    @abstractmethod
    def univariate_analysis(self, df: pd.DataFrame):
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysis):
    def univariate_analysis(self, df: pd.DataFrame):
        # Seleksi kolom numerik
        df = df.select_dtypes(include=['number'])
        numerical_columns = df.columns
        num_columns = len(numerical_columns)
        numerical_rows = (num_columns * 2 + num_columns - 1) // num_columns
        fig, axes = plt.subplots(nrows=numerical_rows, ncols=num_columns, figsize=(20, 5 * numerical_rows))
        axes = axes.flatten()
        for i, column in enumerate(numerical_columns):
            # Plot histogram
            sns.histplot(data=df, x=column, ax=axes[i*2], kde=True)
            axes[i*2].set_title(f"Distribution of {column}")
            # Plot boxplot
            sns.boxplot(data=df, x=column, ax=axes[i*2 + 1])
            axes[i*2 + 1].set_title(f"Boxplot of {column}")
        for j in range(i*2 + 2, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysis):
    def univariate_analysis(self, df: pd.DataFrame):
        df = df.select_dtypes(exclude=['number'])
        categorical_columns = df.columns
        num_columns = len(categorical_columns)
        max_columns_per_row = 2
        categorical_rows = (num_columns + max_columns_per_row - 1) // max_columns_per_row
        fig, axes = plt.subplots(nrows=categorical_rows, ncols=max_columns_per_row, figsize=(20, 5 * categorical_rows))
        axes = axes.flatten()
        for i, column in enumerate(categorical_columns):
            # Plot Barplot
            sns.violinplot(data=df, x=column, ax=axes[i])
            axes[i].set_title(f"Violinplot of {column}")
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

class UnivariateAnalysisFactory:
    def __init__(self, strategy : UnivariateAnalysis):
        self.strategy = strategy

    def analyze(self, df : pd.DataFrame):
        return self.strategy.univariate_analysis(df)

if __name__ == "__main__":
    pass