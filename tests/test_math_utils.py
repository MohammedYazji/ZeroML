import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zeroml.core.math_utils import clip, to_2d_column

def test_clip_zero():
    value = np.array([0.0])

    result = clip(value)

    # Check Result
    assert result[0] > 0.0, "Value should become > 0.0"

def test_to_2d_column():
    # vector - 1d array
    value = np.array([1, 2, 3])

    result = to_2d_column(value)

    assert result.shape[1] == 1, "Should be matrix with one column" 