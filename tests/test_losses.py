import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zeroml.core.losses import MAE, MSE, BinaryCrossEntropy



def test_mse_returns_zero_for_perfect_predictions():
    
    y_true = np.array([[23], [43], [12]])
    y_pred = np.array([[23], [43], [12]])

    loss = MSE()
    result = loss.compute(y_true=y_true, y_pred=y_pred)

    # Check the result
    assert np.allclose(result, 0.0), "Cost must be 0"

def test_mae_returns_zero_for_perfect_predictions():
    
    y_true = np.array([[23], [43], [12]])
    y_pred = np.array([[23], [43], [12]])

    loss = MAE()
    result = loss.compute(y_true=y_true, y_pred=y_pred)

    # Check the result
    assert np.allclose(result, 0.0), "Cost must be 0"
