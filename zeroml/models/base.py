from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract class for all models in the package.

    Every model must implement fit() and predict().
    This enforces a consistent interface across all models.
    """

    @abstractmethod
    def fit(self, X, y):
        """Train the model on data X with labels y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for input X."""
        pass