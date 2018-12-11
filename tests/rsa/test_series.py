import sys
from pathlib import Path

import numpy as np

src_dir = Path(__file__).parent.joinpath("../../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.assessment import RSA_Assessment_Logic


def test_assessment_list(series_logic):
    # verify that the series can identify these assessments
    assert sorted(["dx0", "dx500"]) == sorted(series_logic.assessment_list())


def test_assessment(series_logic):
    assessment_logic = series_logic.assessment("dx0")
    assert isinstance(assessment_logic, RSA_Assessment_Logic)
    assert assessment_logic.name == "dx0"


# DISABLED: This test is disables as the refactoring has split this up in two different calls
# def test_scene_segment_motion(scene_segment_motion):
#     #T, distances =
#     R, t, dists = scene_segment_motion.find_motion()
#     # T_true was calculated in notebooks/Point set registration/ICP.ipynb
#     T_true = np.array([
#         [9.99998631e-01, 1.55049295e-03, -5.78619029e-04, 6.46670881e-01],
#         [-1.55071589e-03, 9.99998724e-01, -3.85042746e-04, 2.03770938e-02],
#         [5.78021284e-04, 3.85939493e-04, 9.99999758e-01, -2.25688157e-02],
#         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
#     ])
#
#     assert np.allclose(
#         R,
#         np.array([
#             [9.99998539e-01,  1.57436586e-03, -6.66467742e-04],
#             [-1.57470583e-03,  9.99998630e-01, -5.09892864e-04],
#             [6.65664072e-04,  5.10941609e-04,  9.99999648e-01]
#         ])
#     )
#
#     assert np.allclose(
#         t,
#         np.array([
#             0.64619549,
#             -0.0086944,
#             -0.05610896
#         ])
#     )

