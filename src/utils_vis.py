import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, r
from sklearn.metrics.pairwise import pairwise_distances

numpy2ri.activate()


def column_reorder(X, return_index=False, on=None, metric="euclidean"):
    """Reorder column of `X` to minimize distance between successive columns.

    Args:
        X: 2-dimensional numpy array
        return_index: also return the reordering
        on: limit the samples considered to reorder the columns
        metric: metric to use between columns

    Returns:
        the matrix `X` with its column reordered so that the transition between
        one column to the next is as smooth as possible.

    """

    # Consider all samples by default
    if on is None:
        on = np.arange(X.shape[0])

    # Select samples used to reorder columns
    X_on = X[on, :]

    # Compute pairwise distance between columns
    dists = pairwise_distances(X_on.T, X_on.T, metric=metric)

    # Use Travel Salesman Problem to find a better column ordering
    r.assign("X", dists)
    result = r(r"""
    library("TSP")
    tsp = as.TSP(dist(X))
    tsp = insert_dummy(tsp, label="dummy")
    order = solve_TSP(tsp)
    cut_tour(order, "dummy")
    """)

    # Zero-based
    result = np.asarray(result) - 1

    if return_index:
        return X[:, result], result
    else:
        return X[:, result]
