"""

Logic related to an RSA assessment, e.g. generating crossing lines tables, transforming all segments' points according to reference segment.

"""

from collections import OrderedDict

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from .dataset import RSA_Dataset
from .projection import RSA_Projection_Logic, Line


class RSA_Assessment_Logic(object):
    def __init__(self, dataset, key_string):
        assert isinstance(dataset, RSA_Dataset), "Provided dataset instance of {} and not RSA_Dataset".format(
            type(dataset))
        self.dataset = dataset
        self.key_string = key_string
        self._idict = dataset.get(key_string)

    def projection_list(self):
        image_names = []
        try:
            image_names = [i["name"] for i in self._idict["projections"]]
        # ignore errors caused by non-existing projection list
        except KeyError:
            self._idict["projections"] = []
        assert len(image_names) <= 2, "More than two projections is not supported"
        return image_names

    def projection(self, name):
        assert name in self.projection_list(), "Projection '{}' not in assessment '{}' in current dataset".format(name,
            self.key_string)
        return RSA_Projection_Logic(self.dataset, "{}.projections.[name={}]".format(self.key_string, name), parent=self)

    def scene_segment_list(self):
        scene_segment_names = [s["name"] for s in self.dataset.get("segments")]

        # strip out items from groups
        scene_segment_names = [s.split("/")[0] for s in scene_segment_names]

        # strip out multiple occurances of groups
        return list(OrderedDict.fromkeys(scene_segment_names))  # thanks to https://stackoverflow.com/a/7961390

    def scene_segment(self, name: str, **kwargs):
        """
        Instantiate an RSA_Scene_Segment object from a segment name

        This instantiates this assessment's scene segment by the given name. If it exists in the dataset under "/segments" but not under "/assessments/[this assessment]/scene_segments", it is instantiated.

        Parameters
        ----------
        name: string
            Name of scene segment. If the segment is the name of a group (e.g. where the images contain segments foo/bar and foo/baz, foo is the group), lines from all images segment in the group are added to the scene segment.

        fill_lines: bool
            Fill the scene segment with lines from image segments. Default: True.

        match_lines: bool
            Match lines from each projection onto one another. Implies fill_lines=True. Default: False. See RSA_Scene_Segment.match_crossing_lines for full documentation.

        Returns
        -------
        Instance of RSA_Scene_Segment.
        """

        assert name in self.scene_segment_list()

        if not "scene_segments" in self._idict.keys():
            self._idict["scene_segments"] = []

        # instantiate scene segment
        ss = RSA_Scene_Segment(self.dataset, "{}.scene_segments.[name={}]".format(self.key_string, name), parent=self)

        # collect lines from image segments with identical name or group
        if kwargs.get("fill_lines", True) or kwargs.get("match_lines", False):
            lines = [[], []]

            projection_names = sorted(
                self.projection_list())  # make sure this is sorted to make a reliable lines_a_idx in scene_segment
            for i in range(2):
                projection = self.projection(projection_names[i])

                for image_segment_name in projection.image_segment_list():
                    if not image_segment_name.split("/")[0] == name:
                        continue

                    image_segment = projection.image_segment(image_segment_name)
                    [lines[i].append(line) for line in projection.points_to_lines(image_segment.points)]
                    # lines[i].append(image.points_to_lines(image_segment.points))

            # set lines of scene segment
            ss.lines_a = lines[0]
            ss.lines_b = lines[1]

        if kwargs.get("match_lines", False):
            ss.match_crossing_lines()

        return ss

    @property
    def name(self):
        return self._idict["name"]


class RSA_Scene_Segment(object):
    # tolerance for termination of search for shortest line between this and another line, set to 1/1000th of suggested limit for MSE for clinical research
    tol = 0.00035

    def __init__(self, dataset, key_string, **kwargs):
        self.dataset = dataset
        self.key_string = key_string

        try:
            self._idict = dataset.get(key_string)

        # check if segment exist under image segments in dataset, create if not
        except KeyError:
            keys = key_string.split(".")
            last_key = keys[-1]
            name = last_key[last_key.find("=") + 1:-1]

            # make sure we are allowed to create this segment
            assert name in kwargs.get("parent").scene_segment_list()

            self._idict = {
                "name": name
            }
            dataset.get(".".join(keys[:-1])).append(self._idict)

        self.lines_a = kwargs.get("lines_a", [])
        self.lines_b = kwargs.get("lines_b", [])

    def name(self):
        return self._idict["name"]

    def min_dist_line(self, line_a, line_b):
        """
        Find the minimum distance from line_a to line_b and returns a line representing this.
        """
        res = minimize(
            lambda lambdas, line_a, line_b: np.linalg.norm(line_a.r(lambdas[0])[:-1] - line_b.r(lambdas[1])[:-1]),
            (0, 0),
            args=(line_a, line_b),
            # bounds=((0, 1), (0, 1)),
            tol=self.tol
        )
        return Line(line_a.r(res.x[0]), line_b.r(res.x[1]))

    def match_crossing_lines(self):
        # TODO: documentation for this method
        # TODO: QC!
        # TODO: save point index to dataset
        crossing_lines = []
        self.dist_map = np.zeros((len(self.lines_a), len(self.lines_b)))

        for i in range(len(self.lines_a)):
            crossing_lines.append([])
            line_a = self.lines_a[i]
            for j in range(len(self.lines_b)):
                line_b = self.lines_b[j]
                crossing_line = self.min_dist_line(line_a, line_b)
                crossing_lines[i].append(crossing_line)
                self.dist_map[i, j] = crossing_line.length

        self.lines_a_idx, self.lines_b_idx = linear_sum_assignment(self.dist_map)
        self._idict["errors"] = self.dist_map[self.lines_a_idx, self.lines_b_idx].tolist()
        self._idict["mean_error"] = float(np.mean(self.dist_map[self.lines_a_idx, self.lines_b_idx]))
        self._idict["mse"] = float(np.mean(np.square(self.dist_map[self.lines_a_idx, self.lines_b_idx])))
        self._idict["points"] = []
        for i in range(len(self.lines_a_idx)):
            self._idict["points"].append(crossing_lines[self.lines_a_idx[i]][self.lines_b_idx[i]].r(0.5)[:-1].tolist())

    @property
    def mse(self):
        return self._idict["mse"]

    @property
    def points(self):
        return self._idict["points"]

# class CrossingLinesTable(object):
#     """
#     Represents one table of crossing lines, e.g. same segment in two projections
#     """
#
#     def __init__(self, segments, projectors):
#         """
#         Instantiate crossing lines table by providing the two segments (one from each projection) and the two corresponding projectors.
#         :param segments:
#         :param projectors:
#         """
#
#         self.segments = segments
#         self.projectors = projectors
#
#         # this var holds crossing lines for each possible point match. Index is lines[index in projection 1][index in projection 2]. Initialised in process_crossing_lines
#         self.lines = []
#         self.dists = np.zeros((
#             len(self.segments[0].points),
#             len(self.segments[1].points)
#         ))
#
#     def points(self):
#         """
#         Get a list of 3D projection points (e.g. x,y,z,w) that are represented by the given lines in the current projection.
#         :return:
#         """
#         pp = np.zeros((len(self.row_idx), 4))
#         for i in range(pp.shape[0]):
#             line = self.lines[self.row_idx[i]][self.col_idx[i]]
#             pp[i, ...] = line.r(0.5)
#         return pp
#
#     def process_crossing_lines(self):
#         self.lines = []
#         lines_1 = self.projectors[0].projection_lines(self.segments[0].points)
#         lines_2 = self.projectors[1].projection_lines(self.segments[1].points)
#
#         for i in range(len(self.segments[0].points)):
#             self.lines.append([])
#             line1 = lines_1[i]
#             for j in range(len(self.segments[1].points)):
#                 line2 = lines_2[j]
#                 crossing_line = line1.min_dist_line(line2)
#                 self.lines[i].append(crossing_line)
#                 self.dists[i, j] = crossing_line.length
#
#     def assign_crossing_lines(self):
#         self.row_idx, self.col_idx = linear_sum_assignment(self.dists)
#         # TODO: add range tests here to account for
#         return self.row_idx, self.col_idx
