import numpy as np
from zeroml.models.base import BaseModel
from zeroml.core.math_utils import to_2d_column

class LinearRegression(BaseModel):
    """
    Linear Regression from scratch.
    ===============================
    Learns the relationships between features X and target y by finding the best weights w and bias b such that:

    ŷ = X.w + b

    Training uses Gradient Descent to minimize MSE cost:

    J = (1/2m) * Σ(y - ŷ)²

    Usage:
        model = LinearRegression(learning_rate=0.01, n_iterations-=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Parameters:
        -----------
            learning_rate: float
                How big a step gradient descent takes each iterations.
                Too large -> overshoots (cost explodes)
                Too small -> learns too slowly
            
            n_iterations: int
                How many times gradient descent updates the weights.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.cost_history = [] # track cost per iteration

    def _initialize_weights(self, n_features):
        """
        Initialize weights to zero before training starts.

        w shape -> (n_features, 1) - weight for each feat
        b -> scaler 0

        Parameters:
        -----------
            n_features: int
        """
        self.w = np.zeros((n_features, 1))
        self.b = 0

    def predict(self, X):
        """
        Compute predictions using current weights.

        ŷ = X.w + b

        Parameters:
        -----------
            X : numpy array of shape (m_samples, n_features) - each datapoint has the whole feats
        """
        if self.w is None:
            raise RuntimeError("Model not trained yet. call fit() first.")

        return X @ self.w + self.b
    
    def fit(self, X, y):
        """
        Train the model using Gradient Descent.
        For now: only initialize weights, without Gradient descent.

        Parameters:
        -----------
            X : numpy array of shape (m_samples, n_features)
            y : numpy array of shape (m_samples, 1)
        """
        # Ensure X will be numpy array to use shape and @
        X = np.array(X, dtype=float)
        # May user pass (m_samples,) as vector
        y = to_2d_column(np.array(y, dtype=float))

        m, n = X.shape
        self._initialize_weights(n)