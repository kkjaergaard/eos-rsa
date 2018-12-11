"""

TODO: Add unit tests for this class

"""

import numpy as np
from scipy import spatial, optimize

from .transformation_3d import Transformation


def find_neighbors(A, B):
    """
    Find nearest neighbors. Outliers are removed. Supports asymmetric sets.

    Parameters
    ----------
    A: np.ndarray
        First pointset of shape (n,3) for n non-homogeneous points or (n,4) for homogeneous points.
    B: np.ndarray
        Second pointset of shape (n,3) for n non-homogeneous points or (n,4) for homogeneous points.

    Returns
    -------
    Indices of A, corresponding indices of B, distances between corresponding pairs
    """

    dist_map = spatial.distance.cdist(A[:, :3], B[:, :3])
    A_idcs, B_idcs = optimize.linear_sum_assignment(dist_map)
    # TODO: add outlier removal here
    return A_idcs, B_idcs, dist_map[A_idcs, B_idcs]


def best_fit_transform(A, B):
    """
    Find best transformation from A to B, asuming correspondence. Moves A to align it onto B.

    Parameters
    ----------
    A: np.ndarray
        First pointset of shape (n,3) for n non-homogeneous points or (n,4) for homogeneous points.
    B: np.ndarray
        Second pointset of shape (n,3) for n non-homogeneous points or (n,4) for homogeneous points.

    Returns
    -------
    (3,3) rotation matrix, (1,3) translation vector
    """

    # translate to origin
    A_centroid = np.mean(A[:, :3], axis=0)
    B_centroid = np.mean(B[:, :3], axis=0)

    Ac = A[:, :3] - A_centroid
    Bc = B[:, :3] - B_centroid

    # rotation matrix
    H = np.dot(Ac.T, Bc)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1  # TODO: verify this step
        R = np.dot(Vt.T, U.T)

    # return rotation matrix R and translation vector t
    return R, B_centroid.T - np.dot(R, A_centroid.T)


def icp(A, B, **kwargs):
    """
    Find movement from A to B using iterative closest point

    Parameters
    ----------
    A: np.ndarray
        First pointset of shape (n,3) for n non-homogeneous points or (n,4) for homogeneous points.
    B: np.ndarray
        Second pointset of shape (n,3) for n non-homogeneous points or (n,4) for homogeneous points.
    iterations: int
        Max number of iterations
    tol: float
        Error convergence tolerance, terminates when this is achieved

    Returns
    -------
    (3,3) rotation matrix, (1,3) translation vector, distances between corresponding pairs, number of iterations
    """

    AA = np.copy(A)

    prev_error = 0

    for i in range(kwargs.get("iterations", 20)):

        # find neighbors
        A_idcs, B_idcs, dists = find_neighbors(AA, B)

        # find rmse
        # rmse = np.sqrt(np.mean(np.square(dists)))
        me = np.mean(dists)
        if abs(prev_error - me) < kwargs.get("tol", 0.00035):
            break

        prev_error = me

        # find best transformation
        R, t = best_fit_transform(AA[A_idcs, :3], B[B_idcs, :3])
        AA = Transformation.f().m(R, add_w=True).tl(t).tf(AA, add_w=True)

    R, t = best_fit_transform(A[A_idcs, :3], B[B_idcs, :3])
    return R, t, dists, i
