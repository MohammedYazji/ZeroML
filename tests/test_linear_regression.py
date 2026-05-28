import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zeroml.models.linear_regression import LinearRegression

def test_predict_before_fit_raises_error():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    model = LinearRegression()

    with pytest.raises(RuntimeError):
        model.predict(X)

def test_fit_learns_perfect_linear_relationship():
    # y = 2x exactly
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])

    model = LinearRegression(learning_rate=0.01, n_iterations=5000)
    model.fit(X, y)

    # w should be close to 2, b should be close to 0
    # matrix
    assert np.allclose(model.w, [[2.0]], atol=0.05), 'w must be almost 2 based on the provided data'
    # scaler 
    assert np.allclose(model.b, 0.0, atol=0.01), 'b must be almost 0 based on the provided data'