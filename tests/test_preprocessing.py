import sys
import os
import pytest
import numpy as np
from zeroml.utils.preprocessing import FeatureScaler

# Cause i will run pytest from this current folder tests/
# so it will not know where is zeroml i will go up one directory to search there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zeroml.utils.preprocessing import FeatureScaler

# === Fixture ===
# Instead of declare the same in every test function
# using fixture we can use them where we want after pass them as parameter
@pytest.fixture
def sample_data():
    """
    Simple dataset with 2 features to test scaling.
    Feature 1: [1, 2, 3, 4, 5]
    Feature 2: [100, 200, 300, 400, 500]
    Widely different ranges - perfect for testing scaling.
    """
    return np.array([
        [1, 100],
        [2, 200],
        [3, 300],
        [4, 400],
        [5, 500],
    ], dtype=float)

@pytest.fixture
def train_test_data():
    """
    Separate train and test sets to verify
    that test data uses training statistics.
    """
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    X_test  = np.array([[10.0], [20.0]])
    return X_train, X_test

# === Max Scale Tests ===
def test_maxscale_range(sample_data):
    """After max scaling all values must be between 0 and 1."""
    scaler = FeatureScaler(method='maxscale')
    X_scaled = scaler.fit_transform(sample_data)

    # Check the result
    assert X_scaled.min() >= 0.0, "Min value should be >= 0"
    assert X_scaled.max() <= 1.0, "Max value should be <= 1"

def test_maxscale_maximum_becomes_one(sample_data):
    """The maximum value in each feature must become exactly 1."""
    scaler = FeatureScaler(method='maxscale')
    X_scaled = scaler.fit_transform(sample_data)

    # Check the result
    # I use allclose cause may be there very small different in values
    assert np.allclose(X_scaled.max(axis=0), 1.0), \
    "Max of each feature should be 1 after max scaling."

# === Mean Normalization Tests ===
def test_meannorm_mean_is_zero(sample_data):
    """After mean normalization the mean of each feature must be almost 0."""
    scaler = FeatureScaler(method='meannorm')
    X_scaled = scaler.fit_transform(sample_data)

    # Check the result
    # also here i use allclose cause may be there very small different between values not exactly 0
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-6)

def test_meannorm_ragne(sample_data):
    """After mean normalization values must be roughly between -1 and 1."""
    scaler = FeatureScaler(method='meannorm')
    X_scaled = scaler.fit_transform(sample_data)

    # Check the result
    assert X_scaled.min() >= -1, "Min should be >= -1"
    assert X_scaled.max() <= 1, "Min should be >= 1"

# === Z-Score Tests ===
def test_zcore_mean_is_zero(sample_data):
    """After z-score the mean of each feature must be almost 0."""
    scaler = FeatureScaler(method='zscore')
    X_scaled = scaler.fit_transform(sample_data)

    # Check the result
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-6), \
    "Mean should be almost 0 after z-score"

def test_zscore_std_is_one(sample_data):
    """After z-score the std of each feature must be almost."""
    scaler = FeatureScaler(method='zscore')
    X_scaled = scaler.fit_transform(sample_data)

    # Check the result
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-6), \
    "Std should be almost 1 after z-score"

# === Data Leakage Test ===
def test_transform_uses_training_stats(train_test_data):
    """
    Test data must be scaled using training statistics.
    Not its own mean/std - that would be data leakage.
    """
    X_train, X_test = train_test_data

    scaler = FeatureScaler(method='zscore')
    scaler.fit(X_train)

    # transform test data using the training statistics
    X_test_scaled = scaler.transform(X_test)

    # manually compute expected result using TRAINING mean and std
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    expected = (X_test - train_mean) / (train_std + 1e-8)

    # Check the result
    assert np.allclose(X_test_scaled, expected), \
    "Test data must be scaled using training statistics to prevent data leakage"

# === Safety Tests ===
def test_transform_before_fit_raises_error():
    """Calling transform() before fit() must raise RunTimeError."""
    scaler = FeatureScaler(method='zscore')

    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))

def test_invalid_method_error():
    """Passing an unknown method must raise ValueError."""
    with pytest.raises(ValueError):
        FeatureScaler(method='someStrangeMethod')