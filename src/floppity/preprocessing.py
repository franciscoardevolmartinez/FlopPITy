import numpy as np

def softclip(x):
    """
    From Vasist+23

    Applies a soft clipping function to the input array.

    This function scales the input values using the formula:
    scaled_flux = x / (1 + abs(x / 100))

    Parameters:
    x (numpy.ndarray): A 2D array where each row represents a different spectrum.

    Returns:
    numpy.ndarray: A 2D array with the soft-clipped values.
    """
    scaled_flux = x / (1 + np.abs(x / 100))
    return scaled_flux

def log(x):
    """
    Computes the base-10 logarithm of the input. If the input is a 2D array, computes the
    logarithm element-wise for each value in the array.

    Parameters:
    x (float, array-like, or 2D array): The input value(s) for which to compute the base-10 
                                        logarithm. Must be positive.

    Returns:
    float, ndarray, or 2D ndarray: The base-10 logarithm of the input value(s).

    Raises:
    ValueError: If any input value(s) are not positive.
    """
    if np.any(x <= 0):
        raise ValueError("All input values must be positive to compute the logarithm.")
    
    return np.log10(x)

def standardize_1v1(x):
    """
    Standardizes the input 2D array by subtracting the mean and dividing by the standard deviation
    for each row. Additionally, appends the mean and standard deviation of each row to the result.

    Parameters:
    x (numpy.ndarray): A 2D array where each row represents a different spectrum.

    Returns:
    numpy.ndarray: A 2D array where each row is the standardized input followed by the mean
                   and standard deviation of that row.
    """
    means = np.mean(x, axis=1, keepdims=True)
    stds = np.std(x, axis=1, keepdims=True)
    standardized = (x - means) / stds
    return np.hstack((standardized, means, stds))

def standardize_global(x):
    """
    Standardizes the input array column-wise.

    This function computes the z-score for each element in the input array `x` 
    by subtracting the mean and dividing by the standard deviation for each column.

    Parameters:
    -----------
    x : numpy.ndarray
        A 2D array where each row represents a different spectrum.

    Returns:
    --------
    numpy.ndarray
        A 2D array with standardized values computed column-wise.

    Notes:
    ------
    - Ensure that the input array `x` is a NumPy array.
    - If any column in `x` has zero standard deviation, this will result in a division by zero.
    """
    column_means = np.mean(x, axis=0, keepdims=True)
    column_stds = np.std(x, axis=0, keepdims=True)
    return (x - column_means) / column_stds

