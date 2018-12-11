"""

Unit testing of Transformation class in rsa.transformation.

Performs some basic pointset transformations in 3D and a few chained transformations.

"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.joinpath("../../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.transformation_3d import *

T = Transformation


def test_scale(default_pointset):
    assert np.allclose(
        T().f().sc([.5, 1.5, 2]).tf(default_pointset),
        np.array([
            [0, 0, 0, 1],
            [.5, 1.5, 2, 1],
            [1, 3, 4, 1]
        ])
    )


def test_translation(default_pointset):
    # print(T().f().tl([.5, -1, 1]).tf(default_pointset))
    assert np.allclose(
        T().f().tl([.5, -1, 1]).tf(default_pointset),
        np.array([
            [.5, -1, 1, 1],
            [1.5, 0, 2, 1],
            [2.5, 1, 3, 1]
        ])
    )


def test_rotation(default_pointset):
    assert np.allclose(
        T().f().rt(np.radians([90, 0, -90])).tf(default_pointset),
        np.array([
            [0, 0, 0, 1],
            [-1, -1, 1, 1],
            [-2, -2, 2, 1]
        ])
    )


def test_sc_rt(default_pointset):
    assert np.allclose(
        T().f().sc([.5, 1, 2]).rt(np.radians([90, 0, -90])).tf(default_pointset),
        np.array([
            [0, 0, 0, 1],
            [-2, -.5, 1, 1],
            [-4, -1, 2, 1]
        ])
    )


def test_rot_tl(default_pointset):
    assert np.allclose(
        T().f().rt(np.radians([90, 0, -90])).tl([1, -1, 0]).tf(default_pointset),
        np.array([
            [1, -1, 0, 1],
            [0, -2, 1, 1],
            [-1, -3, 2, 1]
        ])
    )


def test_rotation_matrix_x():
    angles = np.random.rand(10)
    for angle in angles:
        # rotation around y axis
        assert np.allclose(
            rotation_matrix(z=angle),
            np.array([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0]
            ])
        )

        # rotation around y axis
        assert np.allclose(
            rotation_matrix(y=angle),
            np.array([
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)]
            ])
        )

        # rotate around x axis
        assert np.allclose(
            rotation_matrix(x=angle),
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)]
            ])
        )
