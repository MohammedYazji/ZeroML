import numpy as np
from zeroml.models.base import BaseModel
from zeroml.core.math_utils import to_2d_column
from zeroml.core.losses import MSE

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
        
        Steps each iteration:
            1. predict -> ŷ = X.w + b
            2. Compute cost -> j = MSE(y, ŷ)
            3. Compute gradient -> dj/dŷ = (1/m)(ŷ - y)
            4. update x -> w = w - α . X.T . gradient
            4. update b -> b = b - b - α . Σgradient

        Parameters:
        -----------
            X : numpy array of shape (m, n)
            y : numpy array of shape (m, 1)
        """
        # == Prepare data ==
        # Ensure X will be numpy array to use shape and @
        X = np.array(X, dtype=float)
        # May user pass (m_samples,) as vector not matrix
        y = to_2d_column(np.array(y, dtype=float))

        m, n = X.shape
        self._initialize_weights(n)

        loss = MSE()

        for i in range(self.n_iterations):

            # step 1 - predict
            y_pred = self.predict(X)

            # step 2 - compute cost and save it
            cost = loss.compute(y_true=y, y_pred=y_pred)
            self.cost_history.append(cost)

            # step 3 - compute gradient
            error = loss.gradient(y_true=y, y_pred=y_pred)

            # step 4 - update weights
            #  we need to us X.T to make number of columns in the first matrix to equal the number of rows in the second one
            # dj/dw = dj/dŷ . dŷ/dw - while dj/dŷ is the error we calc - dŷ/dw is X
            dw = X.T @ error
            # dj/db = dj/dŷ . dŷ/db - while dj/dŷ is the error we calc - dŷ/db is 1
            # then use sum cause b affects every prediction
            db = np.sum(error)

            # step 5 - gradient descent step
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        
        # to allow method chaining
        return self