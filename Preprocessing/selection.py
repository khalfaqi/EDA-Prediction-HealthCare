import pandas as pd
from abc import ABC, abstractmethod
from sklearn.feature_selection  import SelectKBest, chi2

class Feature(ABC):
    @abstractmethod
    def feature_select(self, df : pd.DataFrame):
        pass

class FeatSelect(Feature):
    def feature_select(self, df: pd.DataFrame, threshold: float = 0.3):
        df = df.select_dtypes(include=['number'])
        X = df.drop(columns=['Test Results'])  
        y = df['Test Results']
        fs = SelectKBest(score_func=chi2, k='all')
        
        # Fit the selector to the data
        fs.fit(X, y)
        
        # Display and store the feature scores
        scores = fs.scores_
        for feat in range(len(scores)):
            print('Feature %d: %f' % (feat, scores[feat]))
        
        # Apply the threshold to filter important features
        important_features = [X.columns[i] for i in range(len(scores)) if scores[i] > threshold]
        
        # Create a new DataFrame with only the important features
        X_new = X[important_features]
        print("Selected features:", important_features)
        
        # Combine the selected features with the target column
        df_new = pd.concat([X_new, y], axis=1)
        return df_new
    
class SelectionFactory:
    def __init__(self, strategy : Feature):
        self.strategy = strategy

    def feature(self, df : pd.DataFrame):
        return self.strategy.feature_select(df)
    
if __name__ == "__main__":
    pass