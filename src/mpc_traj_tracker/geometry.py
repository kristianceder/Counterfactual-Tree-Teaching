import numpy as np
from scipy import spatial  # type: ignore


def polygon_halfspace_representation(polygon_points:np.ndarray):
    '''Compute the H-representation of a set of points (facet enumeration). \\
    Reference
        Code: https://github.com/d-ming/AR-tools/blob/master/artools/artools.py
    Return
        A: (L x d) array. Each row in A represents hyperplane normal.
        b: (L x 1) array. Each element in b represents the hyperpalne constant bi
    '''
    hull = spatial.ConvexHull(polygon_points)
    hull_center = np.mean(polygon_points[hull.vertices, :], axis=0)  # (1xd) vector
    
    K = hull.simplices
    V = polygon_points - hull_center # perform affine transformation
    A = np.nan * np.empty((K.shape[0], polygon_points.shape[1]))

    rc = 0
    for i in range(K.shape[0]):
        ks = K[i, :]
        F = V[ks, :]
        if np.linalg.matrix_rank(F) == F.shape[0]:
            f = np.ones(F.shape[0])
            A[rc, :] = np.linalg.solve(F, f)
            rc += 1

    A:np.ndarray = A[:rc, :]
    b:np.ndarray = np.dot(A, hull_center.T) + 1.0
    return b.tolist(), A[:,0].tolist(), A[:,1].tolist()
