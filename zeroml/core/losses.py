import numpy as np
from zeroml.core.math_utils import clip

class MSE:
    """
    Mean Squared Error Loss
    =======================
    Used for: Linear Regression (regression problems)

    Formula:
        J = (1/2m) * Σ(ŷ - y)²

    The 1/2 cancels the 2 from the derivative:
        dj/dŷ = (1/m) * (ŷ - y) <- after cancel the 2

    Properties:
        - Penalizes large errors heavily (squaring effect)
        - Differentiable everywhere (safe for gradient descent)
        - Sensitive to outliers
    """

    def compute(self, y_true, y_pred):
        """
        Compute MSE cost.

        Parameters:
        ===========
            y_true : numpy array shape (m, 1) - actual values
            y_pred : numpy array shape (m, 1) - predicted values

        Returns:
        ========
            float : single cost value
        """
        m = len(y_true)
        return (1 / (2 * m)) * (np.sum( y_pred - y_true) ** 2)
    
    def gradient(self, y_true, y_pred):
        """
        Derivative of MSE with respect to predictions.

        dj/dŷ = (1/m) * (ŷ - y)

        This tells gradient descent how wrong each
        prediction is and in which direction to fix it.

        Parameters
        ==========
        y_true : numpy array shape (m, 1)
        y_pred : numpy array shape (m, 1)

        Returns
        ========
        numpy array shape (m, 1)
        """
        # another way to get m
        m = y_true.shape[0]
        return (1 / m) * (y_pred - y_true)
    
class MAE:
    """
    Mean Absolute Error Loss
    ========================
    Used for: Evaluation metric (not for gradient descent)

    Formula:
        j = (1/m) * Σ|ŷ - y|

    Properties:
        - Less sensitive to outliers than MSE
        - Not differentiable at 0 (can't use in gradient descent)
        - More interpretable - error in same unit as target
        e.g. predicting house  prices -> MAE = $5000 off on average
    """

    def compute(self, y_true, y_pred):
        """
        Compute MAE cost.

        Parameters:
        ===========
            y_true : numpy array shape (m, 1)
            y_pred : numpy array shape (m, 1)

        Returns
        =======
        float - single cost value
        """
        m = len(y_true)
        return (1 / m) * np.sum(np.abs(y_pred - y_true))
    
class BinaryCrossEntropy:
    """
    Binary Cross Entropy Loss
    =========================
    used for: Logistic Regression (binary classification)

    Formula:
        j = -(1/m) * Σ[y.log(ŷ) + (1-y).log(1-ŷ)]

    Intuition:
        When y=1: loss = -log(ŷ)
            ŷ -> 1.0 loss -> 0  (confident and correct)
            ŷ -> 0.0 loss -> ∞  (confident and wrong)

        When y=0: loss = -log(1-ŷ)
            ŷ -> 1.0 loss -> ∞  (confident and wrong)
            ŷ -> 0.0 loss -> 0  (confident and correct)

    Note:
        clip() is called before log() to prevent log(0) = -infinity
    """

    def compute(self, y_true, y_pred):
        """
        Compute Binary Cross Entropy cost.

        Parameters:
        ===========
            y_true : numpy array shape (m, 1) - actual labels (0 or 1)
            y_pred : numpy array shape (m, 1) - predicted probabilities

        Returns
        =======
        float - single cost value
        """
        m = len(y_true)
        y_pred = clip(y_pred)
        return -(1 / m) * np.sum(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
    
    def gradient(self, y_true, y_pred):
        """
        Derivative of BCE with respect to predictions.

        dj/dŷ = (1/m) * (-(y/ŷ) + (1-y)/(1-ŷ))

        Parameters
        ==========
        y_true : numpy array shape (m, 1)
        y_pred : numpy array shape (m, 1)

        Returns
        =======
        numpy array shape (m, 1)
        """
        
        m = len(y_true)
        y_pred = clip(y_pred)
        return (1 / m) * (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred))