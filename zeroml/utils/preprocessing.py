import numpy as np

class FeatureScaler:
    """
    Feature Scaling for Machine Learning
    ====================================
    When feature have widely different ranges, gradient descent bounces back and forth and converges very slowly.

    Example: 
        Size (sqft): 100 - 1500
        Bedrooms:    1   - 5

    The cost function become an elongated bowl shape.
    After scaling, it becomes a round bowl and gradient descent goes straight into the minimum.

    Usage:
        scaler = FeatureScaler(method='zscore')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    """

    METHODS = ('maxscale', 'meannorm', 'zscore')

    def __init__(self, method='zscore'):
        """
        Parmaeters
        ----------
        method : str
            'maxscale' -> divide by maximum             range[0, 1]
            'meannorm' -> center then scale by range    range[-1, 1]
            'zscore'   -> center then scale by std      mean=0, std=1
        """
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, you choose '{method}'")
        
        self.method = method
        self._fitted = False

        # those will computed during fit()
        self.mean_ =None
        self.std_ = None
        self.max_ = None
        self.min_ = None

    def fit(self, X):
        """
        Compute and store the scaling statistics from training data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)

        # axis=0 means compute per column (per feature)
        # so each feature gets its own mean, std, max, min
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.max_ = X.max(axis=0)
        self.min_ = X.min(axis=0)

        self._fitted = True
        return self
    
    def transform(self, X):
        """
        Apply scaling using the statistics computed during fit().
        Call this on both training and test data.

        Parameters:
        ----------
            X : numpy array of shape (n_samples, n_features)
        """
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        
        X = np.array(X, dtype=float)

        if self.method == 'maxscale':
            # Formula: x_scaled = x / x_max
            # Result: [0, 1]
            # Problem: sensitive to outliers
            return X / (self.max_ + 1e-8)
        
        elif self.method == 'meannorm':
            # Formula: xn = (x - mean) / (x_max - min)
            # Result: centerd at 0, range almost [-1, 1]
            range_ = self.max_ - self.min_
            return (X - self.mean_) / (range_ + 1e-8)
        
        elif self.method == 'zscore':
            # Formula: xn = (x - mean) / std
            # Result: mean=0, std=1, no fixed range
            # best for both linear/logistic regression and nn
            return (X - self.mean_) / (self.std_ + 1e-8)
        
    def fit_transform(self, X):
        """
        Fit on X then transform X in one step.
        USE ONLY ON TRAINING DATA.

        Parameters:
        -----------
            X : numpy array of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)