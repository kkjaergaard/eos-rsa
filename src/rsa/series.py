import logging
from collections import OrderedDict

import numpy as np

from .assessment import RSA_Assessment_Logic
from .dataset import RSA_Dataset
from .registration import icp
from .transformation_3d import transform, Transformation

T = Transformation


class RSA_Series_Logic(object):
    """
    Logic concerned with an RSA series. Provides methods to set reference segment and calculate motion between the other segments from assessment to assessment.
    """

    def __init__(self, dataset):
        assert isinstance(dataset, RSA_Dataset), "Provided dataset instance of {} and not RSA_Dataset".format(
            type(dataset))
        self.dataset = dataset

    def assessment_list(self):
        return [a["name"] for a in self.dataset.get("assessments")]

    def assessment(self, name):
        alst = self.assessment_list()
        assert name in alst, "Assessment '{}' not in current dataset".format(name)
        return RSA_Assessment_Logic(self.dataset, "assessments.[name={}]".format(name))

    def scene_segment_list(self):
        scene_segment_names = [s["name"] for s in self.dataset.get("segments")]

        # strip out items from groups
        scene_segment_names = [s.split("/")[0] for s in scene_segment_names]

        # strip out multiple occurances of groups
        return list(OrderedDict.fromkeys(scene_segment_names))  # thanks to https://stackoverflow.com/a/7961390


class RSA_Scene_Segment_Motion(object):

    def __init__(self, **kwargs):
        """
        Instantiate scene segment motion object. This object can find motion between two assessments by using a reference segment. If the reference segment is not provided, no

        Parameters
        ----------
        kwargs
        """
        self._from = kwargs.get("_from", None)
        self.to = kwargs.get("to", None)
        self.segment_name = kwargs.get("segment_name", None)
        self.logger = logging.getLogger(__name__)

    def find_motion(self, **kwargs):
        """
        Calculates the motion of this scene segment.

        Given scene segments A and B, motion is calculated from A to B.

        First, movement of the reference segment is found and aplied to A0 to align the two assessments. The A0 and A1 are moved so that A0's centroid is in the global origin; this is done to eliminate translation while rotating A0. Then, rotation matrix R and translation vector t are identified.

        Parameters
        ----------
        local: boolean, default False
            If true, returns rotation matrix and translation vector relative to the first (_from) segment's centroid, otherwise returns relative to the global origin.

        T_ref: np.array
            Reference transformation to apply to first (_from) segment

        Returns
        -------
        3x3 rotation matrix, 1x3 translation vector, distances between corresponding points after alignment, number of iterations
        """

        assert isinstance(self._from, RSA_Assessment_Logic), "Assessment to find motion from is not set"
        assert isinstance(self.to, RSA_Assessment_Logic), "Assessment to find motion to is not set"

        local = kwargs.get("local", False)
        T_ref = kwargs.get("T_ref", None)

        A = np.array(self._from.scene_segment(self.segment_name, match_lines=True).points)

        # apply reference transformation to A (default to identity matrix, e.g. no transformation)
        if T_ref is not None:
            assert isinstance(T_ref, np.ndarray), "Please provide a Numpy array"
            assert T_ref.shape == (
            4, 4), "Please provide compatible (4x4) transformation matrix as reference transformation"

            A = transform(A, T_ref)

        B = np.array(self.to.scene_segment(self.segment_name, match_lines=True).points)

        # move A and B to the centroid of A to get rotation and translation relative to A's centroid
        if local:
            A_centroid = np.mean(A[:, :3], axis=0)
            A = A[:, :3] - A_centroid
            B = B[:, :3] - A_centroid

        R, t, dists, i = icp(A, B)
        return R, t, dists


# class RSA_Scene_Segment_Motion(object):
#
#     def __init__(self, **kwargs):
#         """
#         Instantiate scene segment motion object. This object can find motion between two assessments by using a reference segment. If the reference segment is not provided, no
#
#         Parameters
#         ----------
#         kwargs
#         """
#         self._from = kwargs.get("_from", None)
#         self.to = kwargs.get("to", None)
#         self.segment_name = kwargs.get("segment_name", None)
#         self.refsegment_name = kwargs.get("refsegment_name", None)
#         self.logger = logging.getLogger(__name__)
#
#     def find_motion(self):
#         """
#         Calculates the motion of the scene segment.
#
#         Given scene segments A0 and A1, motion is calculated from A0 to A1.
#
#         First, movement of the reference segment is found and aplied to A0 to align the two assessments. The A0 and A1 are moved so that A0's centroid is in the global origin; this is done to eliminate translation while rotating A0. Then, rotation matrix R and translation vector t are identified.
#
#         Returns
#         -------
#         3x3 rotation matrix, 1x3 translation vector, distances between corresponding points after alignment, number of iterations
#         """
#
#         assert isinstance(self._from, RSA_Assessment_Logic), "Assessment to find motion from is not set"
#         assert isinstance(self.to, RSA_Assessment_Logic), "Assessment to find motion to is not set"
#
#         # find R and t for reference segment, then apply this to the first moving segment (A0)
#         Ref0 = np.array(self._from.scene_segment(self.refsegment_name, match_lines=True).points)
#         Ref1 = np.array(self.to.scene_segment(self.refsegment_name, match_lines=True).points)
#         R, t, dists, i = icp(Ref0, Ref1)
#
#         A0 = np.array(self._from.scene_segment(self.segment_name, match_lines=True).points)
#         A1 = np.array(self.to.scene_segment(self.segment_name, match_lines=True).points)
#         A0 = T.f().m(R, add_w=True).tl(t).tf(A0, add_w=True)
#
#         # move A0 so that it's centroid is in the global origo
#         # this means that A0 does not translate while being rotated
#         A0_centroid = np.mean(A0[:, :3], axis=0)
#         A0 = A0[:, :3] - A0_centroid
#         A1 = A1[:, :3] - A0_centroid
#
#         return icp(A0[:, :3], A1[:, :3])
