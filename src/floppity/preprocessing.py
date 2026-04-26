import numpy as np


class PCATransformer:
    """Small PCA transformer for 2D retrieval spectra."""

    def __init__(self, n_components):
        if int(n_components) <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.requested_components = int(n_components)
        self.n_components = None
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    @property
    def fitted(self):
        return self.components_ is not None

    def fit(self, x):
        """Fit PCA on rows of a 2D array or tensor."""
        x_array = self._as_numpy(x)
        if x_array.ndim != 2:
            raise ValueError("PCA input must be a 2D array.")

        max_components = min(x_array.shape)
        if self.requested_components > max_components:
            print(
                "pca_components is larger than the available matrix rank; "
                f"using {max_components} components instead."
            )
        self.n_components = min(self.requested_components, max_components)

        self.mean_ = np.mean(x_array, axis=0, keepdims=True)
        centered = x_array - self.mean_
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vh[: self.n_components]

        total_variance = np.sum(singular_values ** 2)
        if total_variance > 0:
            self.explained_variance_ratio_ = (
                singular_values[: self.n_components] ** 2 / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self

    def transform(self, x):
        """Project rows of a 2D array or tensor into fitted PCA space."""
        if not self.fitted:
            raise RuntimeError("PCA transformer must be fitted before transform.")

        x_array = self._as_numpy(x)
        transformed = (x_array - self.mean_) @ self.components_.T
        return self._like_input(transformed, x)

    def inverse_transform(self, x):
        """Map PCA components back to the original feature space."""
        if not self.fitted:
            raise RuntimeError("PCA transformer must be fitted before inverse_transform.")

        x_array = self._as_numpy(x)
        reconstructed = x_array @ self.components_ + self.mean_
        return self._like_input(reconstructed, x)

    @staticmethod
    def _as_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _like_input(x_array, original):
        if hasattr(original, "detach"):
            import torch

            return torch.as_tensor(
                x_array,
                dtype=original.dtype,
                device=original.device,
            )
        return x_array


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
