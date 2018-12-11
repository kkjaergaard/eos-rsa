import logging

import numpy as np


class Transformation(object):
    """
    Utility class to provide easier transformations on point clouds. Call the functions in the order you want them applied to the point cloud. Each function returns the object itself to they can be chained.

    Usage: T.f().sc(0.5, 0.5, 0.5).rt(...).tl(...).tf(pp, ...)

    Usual order: Scale, then rotate, then translate. You usually would not need to use any other order than this.

    See https://blender.stackexchange.com/a/1825 for order and matrix notation of transformations
    """

    def __init__(self):
        self.N = []
        self.logger = logging.getLogger(__name__)

    def sc(self, *args):
        """
        See scale.
        """
        return self.scale(*args)

    def scale(self, *args):
        if len(args) == 1:
            args = args[0]
        assert len(args) == 3, "Please provide scaling for all three axis"

        self.logger.debug("Scaling by ({}, {}, {})".format(*args))

        self.N.append(np.array([
            [args[0], 0, 0, 0],
            [0, args[1], 0, 0],
            [0, 0, args[2], 0],
            [0, 0, 0, 1]
        ]))
        return self

    def rt(self, *args):
        """
        See rotate.
        """
        return self.rotate(*args)

    def rotate(self, *args):
        """
        Rotate around x, y, and z axis. Provide either list of rotation angles or three arguments. Unit is radians.

        If you want another order than x, y, z, you must call this function multiple times and set rotation around unwanted axes to zero.
        """
        if len(args) == 1:
            args = args[0]
        assert len(args) == 3, "Please provide rotation for all three axis"

        self.logger.debug("Rotating by ({}, {}, {})".format(*args))

        M = np.identity(4)
        M[:3, :3] = rotation_matrix(x=args[0], y=args[1], z=args[2])
        self.N.append(M)

        return self

    def tl(self, *args):
        """
        See translate
        """
        return self.translate(*args)

    def translate(self, *args):
        """
        Translate by x, y, z. Provide either translation vector or list of components.

        Parameters
        ----------
        (x,y,z) or x, y, z
            Add translation by the given coordinates. Either a 3-tuple or three separate arguments.

        Returns
        -------
        self
            Returns the class itself for easy chaining.
        """
        if len(args) == 1:
            args = args[0]
        assert len(args) == 3, "Please provide translation for all three axis"

        self.logger.debug("Translating by ({}, {}, {})".format(*args))

        self.N.append(np.array([
            [1.0, 0.0, 0.0, args[0]],
            [0.0, 1.0, 0.0, args[1]],
            [0.0, 0.0, 1.0, args[2]],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        return self

    def tf(self, pp, **kwargs):
        """
        See transform
        """
        return self.transform(pp, **kwargs)

    def add_matrix(self, M, **kwargs):
        """
        Add a rotation transformation matrix

        Adds the matrix to the list of transformation matrices that are applied to the point set. Useful if you have the matrix.

        Parameters
        ----------
        M: numpy.ndarray
            A (4, 4) transformation matrix

        Returns
        -------
        Returns the object itself for chaining
        """

        if kwargs.get("add_w", False) and M.shape == (3, 3):
            Mh = np.identity(4)
            Mh[:3, :3] = M
            M = Mh

        assert M.shape == (4, 4), "Please provide a (4, 4) rotation matrix"

        self.N.append(M)
        return self

    def m(self, M, **kwargs):
        """
        See add_matrix
        """
        return self.add_matrix(M, **kwargs)

    def transform(self, pp, **kwargs):
        """
        Apply transformation to pointset. Pointset must be a numpy array of shape (n,3) or (n,4) where n is number of point in the set.

        Parameters
        ----------
        pp: ndarray
            List of points to transform, must be (n,3) or (n,4) matrix
        invert=False: boolean
            Invert the transformation, e.g. apply the inverse transformation
        retm=False: boolean
            Return the transformation matrix
        add_w=False: boolean
            Add w dimension for homogenity if it is missing, useful shortcut if your coordinates are not homogeneous

        Returns
        -------
        ndarray OR (ndarray, ndarray)
            Returns transformed points or transformed points and transformation matrix
        """
        assert len(self.N) > 0, "No transformations to apply"
        assert len(pp.shape) == 2, "pp does not have compatible shape, please provide (n,4) matrix"

        # allow to add w for homogenity, but fail if we are not allowed and pp is not homogeneous
        if pp.shape[1] == 3 and kwargs.get("add_w", False):
            pp = np.append(
                pp,
                np.ones([pp.shape[0], 1]),
                axis=1
            )
        assert pp.shape[1] == 4, "pp does not have compatible shape, all points needs 4 coordinates (x,y,z,w)"

        if len(self.N) == 1:
            M = self.N[0]
        else:
            M = np.linalg.multi_dot(list(reversed(self.N)))

        if kwargs.get("invert", False):
            M = np.linalg.inv(M)

        if kwargs.get("retm", False):
            return np.einsum("ij,kj->ik", pp, M), M
        return np.einsum("ij,kj->ik", pp, M)

    @classmethod
    def factory(cls):
        return cls()

    @classmethod
    def f(cls):
        return cls()


def rotation_matrix(**kwargs):
    """
    Construct a 4x4 rotation matrix from axis-angles. Provide axis (x, y, or z) and angle (radians) as parameters.

    One matrix is constructed for each axis, then these matrices are dotted (order: z.y.x) to produce one final matrix.

    Parameters
    ----------
    x: float, default 0
        Rotation around the x axis.
    y: float, default 0
        Rotation around the y axis.
    z: float, default 0
        Rotation around the z axis.

    Returns
    -------
    4x4 rotation matrix of type np.ndarray
    """

    assert len(set(kwargs.keys()).intersection(
        ["x", "y", "z"])) > 0, "No rotation axes provided, please supply at least one axis (x, y, or z)"

    x = kwargs.get("x", 0)
    y = kwargs.get("y", 0)
    z = kwargs.get("z", 0)

    return np.linalg.multi_dot([
        # rotation around z axis
        np.array([
            [np.cos(z), -np.sin(z), 0.0],
            [np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0]
        ]),
        # rotation around y axis
        np.array([
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)]
        ]),
        # rotate around x axis
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)]
        ])
    ])


def transform(P: np.ndarray, M: np.ndarray):
    """
    Apply a transformation given by a (3,3) or (4,4) matrix M to a pointset P of shape (n,3) or (n,4) (non-homogeneous or homogeneous, resp.)

    Parameters
    ----------
    P: np.ndarray
        Pointset. Rows are points, cols are dimensions. Must have 3 (non-homogeneous) or 4 (homogeneous) dimensions.
    M: np.ndarray
        Transformation matrix, must be (3,3) or (4,4) matrix.

    Returns
    -------
    Transformed (homogeneous) pointset
    """

    assert 3 <= P.shape[
        1] <= 4, "Please provide compatible pointset shape, must have 3 (non-homogeneous) or 4 (homogeneous) dimensions"
    assert M.shape == (4, 4) or M.shape == (3, 3), "Please provide a (3,3) or (4,4) transformation matrix"

    if P.shape[1] == 3:
        P = np.append(
            P,
            np.ones([P.shape[0], 1]),
            axis=1
        )

    if M.shape == (3, 3):
        I = np.identity(4)
        I[:3, :3] = M
        M = I

    return np.einsum("ij,kj->ik", P, M)


def rotation_matrix_to_axis_angle(R: np.array):
    raise NotImplementedError
