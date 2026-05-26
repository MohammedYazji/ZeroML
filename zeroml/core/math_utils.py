import numpy as np

def clip(x,epsilon=1e-12):
    """
    Clip values to prevent log(0) = -infinity

    Called before any log() operation in loss function.
    Forces values into range [epsilon, 1 - epsilon].

    Example: 
        clip(0.0) -> 1e-12
        clip(1.0) -> 1 - 1e-12
        clip(0.7) -> 0.7 

    Parameters:
    -----------
        x : numpy array of predication
        epsilon : very small value. Defaults to 1e-12.
    """
    return np.clip(x, epsilon, 1 - epsilon)

def to_2d_column(x):
    """
    Ensure a vector is always shape (n, 1) not (n,).

    This prevents silent shape bugs in matrix multiplication

    Parameters:
    -----------
        x : numpy array
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x