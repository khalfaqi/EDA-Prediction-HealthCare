from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluate(ABC):
    @abstractmethod
    def evaluate(self, df: pd.DataFrame, y_pred):
        pass

class ConfusionMatrixEvaluator(Evaluate):
    def evaluate(self, df: pd.DataFrame, y_pred):
        cm = confusion_matrix(df, y_pred)
        accuracy = accuracy_score(df, y_pred)
        
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.2f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        return cm, accuracy
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.show()

class AccuracyEvaluator(Evaluate):
    def evaluate(self, df: pd.DataFrame, y_pred):
        
        # Compute precision, recall, and F1-score
        precision = precision_score(df, y_pred, average='weighted')  
        recall = recall_score(df, y_pred, average='weighted')
        f1 = f1_score(df, y_pred, average='weighted')
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        
        return precision, recall, f1

class EvaluateFactory:
    def __init__(self, evaluation: Evaluate):
        self.evaluation = evaluation

    def evaluate(self, df: pd.DataFrame, y_pred):
        return self.evaluation.evaluate(df, y_pred)

if __name__ == "__main__":
    pass
