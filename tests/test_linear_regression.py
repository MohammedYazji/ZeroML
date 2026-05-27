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

