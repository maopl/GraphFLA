import numpy as np
from typing import Dict

def mixed_distance(
        X: np.ndarray, 
        ref_vec: np.ndarray, 
        data_types: Dict[int, str],
    ) -> np.ndarray:
        """
        Calculate the mixed distance between each row of matrix `X` and a reference vector `ref_vec`.
        
        The function uses Hamming distance for categorical (including boolean) variables and 
        Manhattan distance for ordinal variables.
        
        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array where each row represents an instance and columns correspond to variables.
            Shape: (n_samples, n_features)
            
        ref_vec : np.ndarray
            A 1D numpy array representing the reference vector, containing values for each feature
            in the dataset. Should match the number of features (columns) in `X`.
            Shape: (n_features,)
        
        data_types : Dict[int, str]
            A dictionary mapping column indices in `X` and `ref_vec` to their respective data types
            ('categorical', 'boolean', 'ordinal').
        
        Returns
        -------
        np.ndarray
            A 1D numpy array of distances between each row in `X` and the `ref_vec`.
            Shape: (n_samples,)
        
        Examples
        --------
        >>> X = np.array([[0, 2, 1], [1, 3, 0], [0, 1, 1]])
        >>> ref_vec = np.array([0, 2, 1])
        >>> data_types = {0: 'categorical', 1: 'ordinal', 2: 'boolean'}
        >>> distances = _mixed_distance(X, ref_vec, data_types)
        >>> print(distances)
        [0 2 1]
        """

        total_distance = np.zeros(X.shape[0])
        
        cat_indices = [index for index, dtype in data_types.items() if dtype == "categorical" or dtype == "boolean"]
        ord_indices = [index for index, dtype in data_types.items() if dtype == "ordinal"]
        
        if cat_indices:
            X_cat = X[:, cat_indices]
            ref_vec_cat = ref_vec[cat_indices]
            hamming_dist = np.sum(X_cat != ref_vec_cat, axis=1)
            total_distance += hamming_dist
        
        if ord_indices:
            X_ord = X[:, ord_indices]
            ref_vec_ord = ref_vec[ord_indices]
            manhattan_dist = np.sum(np.abs(X_ord - ref_vec_ord), axis=1)
            total_distance += manhattan_dist
        
        return total_distance