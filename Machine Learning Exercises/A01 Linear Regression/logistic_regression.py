import numpy as np


def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 1
    # interaction feature
    x1_x2 = (X_data[:, 0] * X_data[:, 1])[:, np.newaxis]
    # square
    x1_sq = X_data[:, 0:1]**2  # [:, 0:1] to keep it as a 2D array
    x2_sq = X_data[:, 1:2]**2
    additional_features = np.concatenate((x1_x2, x1_sq, x2_sq), axis=1)

    X = np.concatenate((X_data, additional_features), axis=1)

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2
    x1_sq = X_data[:, 0:1]**2  # [:, 0:1] to keep it as a 2D array
    x2_sq = X_data[:, 1:2]**2
    additional_features = np.concatenate((x1_sq, x2_sq), axis=1)

    X = np.concatenate((X_data, additional_features), axis=1)

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    # square
    x1_sq = X_data[:, 0:1]**2
    x2_sq = X_data[:, 1:2]**2
    # cubic
    x1_cub = X_data[:, 0:1]**3
    x2_cub = X_data[:, 1:2]**3
    # interaction - reshape (x1 * x2) from (N,) to (N, 1)
    x1_x2 = (X_data[:, 0] * X_data[:, 1])[:, np.newaxis]
    # more flexibility
    x1_sq_x2 = x1_sq * X_data[:, 1:2]
    x2_sq_x1 = x2_sq * X_data[:, 0:1]
    # polynomial
    x1_4 = X_data[:, 0:1]**4
    x2_4 = X_data[:, 1:2]**4
    x1_5 = X_data[:, 0:1]**5
    x2_5 = X_data[:, 1:2]**5
    
    additional_features = np.concatenate((x1_sq, x2_sq, x1_cub, x2_cub, x1_x2, x1_sq_x2, x2_sq_x1, x1_4, x2_4, x1_5, x2_5), axis=1)

    X = np.concatenate((X_data, additional_features), axis=1)

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}
