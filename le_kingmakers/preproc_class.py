from sklearn.base import TransformerMixin, BaseEstimator

from preproc_text import process_tweets


class TextPreprocess(TransformerMixin, BaseEstimator): 
# TransformerMixin generates a fit_transform method from fit and transform
# BaseEstimator generates get_params and set_params methods
    
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.apply(process_tweets)
        # Return result as dataframe for integration into ColumnTransformer
        return X_transformed