import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud as WC

class MedConDataVisualization(ABC):
    @abstractmethod
    def visualize(self):
        pass

class WordCloud(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df = df['Medical Condition']
        wordcloud = WC().generate(' '.join(df.astype(str)))  
        plt.figure(figsize=(15, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

class MedicalCondition(MedConDataVisualization):
    def visualize(self, df : pd.DataFrame):
        df = df[['Gender', 'Medical Condition', 'Age Group']]
        numerical_columns = df.columns
        num_columns = len(numerical_columns)
        numerical_rows = (num_columns + num_columns - 1) // num_columns
        fig, axes = plt.subplots(nrows=numerical_rows, ncols=num_columns, figsize=(20, 5 * numerical_rows))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            sns.countplot(data=df, x=column, ax=axes[i])
            axes[i].set_title(f"Countplot of {column}")
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

class Cancer(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df_smokers = df[df['Medical Condition'] == 'Cancer']
        
        cat = sns.catplot(data=df_smokers, 
                    x='Age Group', 
                    hue='Gender', 
                    kind='count', 
                    height=4, 
                    aspect=2)
        cat._legend.remove() #remove default legend
        cat.add_legend(title='Gender',loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.title('Cancer Distribution by Age Group and Gender')
        cat.set_axis_labels('Age Group', 'Count of Medical Condition')
        plt.tight_layout()
        plt.show()

class Arthritis(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df_smokers = df[df['Medical Condition'] == 'Arthritis']
        
        cat = sns.catplot(data=df_smokers, 
                    x='Age Group', 
                    hue='Gender', 
                    kind='count', 
                    height=4, 
                    aspect=2)
        cat._legend.remove() 
        cat.add_legend(title='Gender',loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.title('Arthritis Distribution by Age Group and Gender')
        cat.set_axis_labels('Age Group', 'Count of Medical Condition')
        plt.tight_layout()
        plt.show()

class Diabetes(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df_smokers = df[df['Medical Condition'] == 'Diabetes']
        
        cat = sns.catplot(data=df_smokers, 
                    x='Age Group', 
                    hue='Gender', 
                    kind='count', 
                    height=4, 
                    aspect=2)
        cat._legend.remove() 
        cat.add_legend(title='Gender',loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.title('Diabetes Distribution by Age Group and Gender')
        cat.set_axis_labels('Age Group', 'Count of Medical Condition')
        plt.tight_layout()
        plt.show()

class Hypertension(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df_smokers = df[df['Medical Condition'] == 'Hypertension']
        
        cat = sns.catplot(data=df_smokers, 
                    x='Age Group', 
                    hue='Gender', 
                    kind='count', 
                    height=4, 
                    aspect=2)
        cat._legend.remove() 
        cat.add_legend(title='Gender',loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.title('Hypertension Distribution by Age Group and Gender')
        cat.set_axis_labels('Age Group', 'Count of Medical Condition')
        plt.tight_layout()
        plt.show()

class Obesity(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df_smokers = df[df['Medical Condition'] == 'Obesity']
        
        cat = sns.catplot(data=df_smokers, 
                    x='Age Group', 
                    hue='Gender', 
                    kind='count', 
                    height=4, 
                    aspect=2)
        cat._legend.remove() 
        cat.add_legend(title='Gender',loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.title('Obesity Distribution by Age Group and Gender')
        cat.set_axis_labels('Age Group', 'Count of Medical Condition')
        plt.tight_layout()
        plt.show()

class Asthma(MedConDataVisualization):
    def visualize(self, df: pd.DataFrame):
        df_smokers = df[df['Medical Condition'] == 'Asthma']
        
        cat = sns.catplot(data=df_smokers, 
                    x='Age Group', 
                    hue='Gender', 
                    kind='count', 
                    height=4, 
                    aspect=2)
        cat._legend.remove() 
        cat.add_legend(title='Gender',loc='upper left', bbox_to_anchor=(1, 1)) 
        plt.title('Asthma Distribution by Age Group and Gender')
        cat.set_axis_labels('Age Group', 'Count of Medical Condition')
        plt.tight_layout()
        plt.show()

class MedConVisualizationFactory:
    def __init__(self, strategy: MedConDataVisualization):
        self.strategy = strategy

    def analyze(self, df: pd.DataFrame):
        return self.strategy.visualize(df)
    
if __name__ == "__main__":
    pass
