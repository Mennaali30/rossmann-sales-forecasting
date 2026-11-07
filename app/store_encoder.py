import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class StoreMeanEncoder(BaseEstimator, TransformerMixin):
    """Mean log-sales per store, learned from the training data only."""
    def fit(self, X, y):
        s = pd.Series(np.asarray(y), index=X["Store"].values)
        self.means_ = s.groupby(level=0).mean()
        self.global_ = float(np.mean(y))
        return self

    def transform(self, X):
        return X["Store"].map(self.means_).fillna(self.global_).to_frame("StoreMeanLogSales")
