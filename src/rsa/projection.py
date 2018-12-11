"""

Logic related to RSA images, e.g. handling of segments, lines (from emitter to image plane) e.g.


# Short introduction to image segments

In brief, an image segment is a set of points on the image that translate to a set of lines into the scene. Any type of segment can be represented as a set of points like this:

* Bead segments: the points are just the beads
* Sphere segments (femoral head, acetabular cup): the point is the center as this is what is of interest
* Other modelled shapes (e.g. knee components): the points are predetermined points in the model

The points are translated to lines in the scene, and lines from two images are combined to identify the segment's points in scene space. Each type of segment has its own logic and UI, but they all produce a set of points on the image that can translate to lines in the scene.

"""

import logging
import math
import re

import numpy as np
import pydicom as dcm
from scipy import optimize
from sklearn import metrics

from .transformation_3d import Transformation


class BeadCenterAdjuster(object):
    METHOD_SINC = "sinc"
    METHOD_SIGMOID = "sigmoid"
    R_MAX = 3.0

    def __init__(self, img, x, y, **kwargs):
        self.img = img
        self.x = x
        self.y = y
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    @classmethod
    def factory(cls, img, x, y, method, **kwargs):
        if method == cls.METHOD_SINC:
            return SincBeadCenterAdjuster(img, x, y, **kwargs)

        if method == cls.METHOD_SIGMOID:
            return SigmoidBeadCenterAdjuster(img, x, y, **kwargs)

        raise NotImplementedError

    def reduce_img_to_polar_radius(self, img: np.ndarray, x: float, y: float, **kwargs):
        """
        Convert a 2d image in cartesian coordinates to polar coordinates and drop angle.

        This function convert the image from 2d cartesian to 1d radius-only polar image.

        Parameters
        ----------
        img: ndarray
            Image to convert
        x: float
            x coordinate to use as origin for cartesian to polar conversion
        y: float
            y coordinate to use as origin for cartesian to polar conversion
        r_max: float
            Drop pixels with a radius larger than this value, optional

        Returns
        -------
        ndarray of shape (2,) where (0,x) describes radius and (1,x) the corresponding pixel value
        """
        X, Y = np.meshgrid(
            np.arange(img.shape[1]),
            np.arange(img.shape[0])
        )
        R = np.sqrt(
            np.square(X - x) +
            np.square(Y - y)
        )

        V = np.copy(img)

        if kwargs.get("r_max", 0.0) > 0.0:
            Xr, Yr = np.where(R <= kwargs.get("r_max"))
            R = R[Xr, Yr]
            V = img[Xr, Yr]

        if kwargs.get("normalize", False):
            V = V - np.min(V)
            V = V / np.max(V)

        return [R.flatten(), V.flatten()]


class SincBeadCenterAdjuster(BeadCenterAdjuster):
    P0_margin = 2.0

    def __init__(self, img, x, y, **kwargs):
        super().__init__(img, x, y, **kwargs)
        self.canvas_size = np.array([15, 15])  # (x,y)
        self.offset = (np.array([x, y]) - self.canvas_size / 2).astype(int)  # (x,y)
        self.P0 = np.array([x, y]) - self.offset

        self.img_window = img[
        self.offset[1]: self.offset[1] + self.canvas_size[1],
        self.offset[0]: self.offset[0] + self.canvas_size[0]
        ]

    def model_sinc(self, R, V_true, full_output=False):
        """
        Find parametes for the sinc model by minimizing MSE from model to observation

        Parameters
        ----------
        full_output: boolean
            If true all output from scipy.optimize.minimize is returned, if false only OptimizationResult.x is returned

        Returns
        -------
        List of optimal parameters or full OptimizationResult object depending on method parameters
        """
        min_amplitude = V_true.max()
        if min_amplitude < 25000.0:
            min_amplitude = 25000.0
        amplitude_range = [
            min_amplitude,
            1.2 * min_amplitude
        ]
        period_range = [2.0, 4.5]
        res = optimize.minimize(
            lambda P_inner, R_inner, V_true_inner: metrics.mean_squared_error(
                V_true_inner,
                P_inner[0] * np.sinc(R_inner / P_inner[1])
            ),
            (
                np.mean(amplitude_range),
                np.mean(period_range)
            ),
            args=(
                R, V_true,
            ),
            bounds=(
                amplitude_range,
                period_range
            )
        )
        if full_output:
            return res
        return res.x

    def sinc_cost_at_xy(self, P, img_window, r_max: float):
        assert len(P) == 2, "Please provide exactly 2 coordinates"
        assert img_window.ndim == 2, "Please provide img_window with two dimensions"

        """
        Define an optimal maximum cost. When approaching the target, it is observed that cost
        can increase rapidly before decreasing to the global minimum. By defining a
        maximum cost, this "cost barrier" is eliminated. Only side-effect: we need to
        get close to the global minimum before we get a gradient for the minimizer.

        This value has been observed to cap cost except exactly at the target.
        """
        max_cost = 7e7

        # if pixel value is less than what we expect from a peak, just quit here
        # if img_window[int(round(P[1])), int(round(P[0]))] < 20000.0:
        #    return max_cost

        R, V_true = self.reduce_img_to_polar_radius(img_window, P[0], P[1], r_max)
        res = self.model_sinc(R, V_true, full_output=True)

        if res.fun > max_cost:
            return max_cost
        return res.fun

    def adjust(self):
        initial_simplex = np.array([
            [self.P0[0], self.P0[1] - self.P0_margin],
            [self.P0[0] - self.P0_margin * math.cos(math.pi / 6), self.P0[1] + self.P0_margin * math.sin(math.pi / 6)],
            [self.P0[0] + self.P0_margin * math.cos(math.pi / 6), self.P0[1] + self.P0_margin * math.sin(math.pi / 6)],
        ])
        res = optimize.minimize(
            self.sinc_cost_at_xy,
            x0=self.P0,
            args=(
                self.img_window,
                self.kwargs.get("r_max", self.R_MAX),
            ),
            method="Nelder-Mead",
            options={
                "disp": True,
                "initial_simplex": initial_simplex,
                "maxiter": 25
            }
        )
        return np.array(res.x) + self.offset


class SigmoidBeadCenterAdjuster(BeadCenterAdjuster):
    P0_margin = 2.0

    def __init__(self, img, x, y, **kwargs):
        super().__init__(img, x, y, **kwargs)

        self.canvas_size = np.array([15, 15])  # (x,y)
        self.offset = (np.array([x, y]) - self.canvas_size / 2).astype(int)  # (x,y)
        self.P0 = np.array([x, y]) - self.offset
        self.r_max = 2.0  # change if you want or override using **kwargs

        # blur using (7,7) gaussian kernel
        self.img_window = img[
        self.offset[1]: self.offset[1] + self.canvas_size[1],
        self.offset[0]: self.offset[0] + self.canvas_size[0]
        ]

    def model_sigmoid(self, R: np.ndarray, V_true: np.ndarray, r_max: float, full_output=False):
        """
        Find parametes for the sigmoid model by minimizing MSE from model to observation

        Parameters
        ----------
        full_output: boolean
            If true all output from scipy.optimize.minimize is returned, if false only OptimizationResult.x is returned

        Returns
        -------
        List of optimal parameters or full OptimizationResult object depending on method parameters
        """

        # see notebook for parameter explanation
        amplitude_range = [1.0, 1.5]
        slope_range = [0.0, 3.0]

        res = optimize.minimize(
            lambda P_inner, R_inner, V_true_inner, r_max_inner: metrics.mean_squared_error(
                V_true_inner,
                P_inner[0] / (
                    1 + np.exp(
                    P_inner[1] * R_inner - (P_inner[1] * r_max_inner / 2)
                )
                )
            ),
            (
                np.mean(amplitude_range),
                np.mean(slope_range)
            ),
            args=(
                R, V_true, r_max,
            ),
            bounds=(
                amplitude_range,
                slope_range
            )
        )
        if full_output:
            return res
        return res.x

    def sigmoid_cost_at_xy(self, P, img_window, r_max: float):
        assert len(P) == 2, "Please provide exactly 2 coordinates"
        assert img_window.ndim == 2, "Please provide img_window with two dimensions"

        """
        Define an optimal maximum cost. When approaching the target, it is observed that cost
        can increase rapidly before decreasing to the global minimum. By defining a
        maximum cost, this "cost barrier" is eliminated. Only side-effect: we need to
        get close to the global minimum before we get a gradient for the minimizer.

        This value has been observed to cap cost except exactly at the target.
        """

        # see notebook for parameter explanation
        max_cost = 0.05  # 0.05 for r_max = 2.0, 0.03 for r_max = 3.0

        R, V_true = self.reduce_img_to_polar_radius(img_window, P[0], P[1], r_max=r_max, normalize=True)
        res = self.model_sigmoid(R, V_true, r_max, full_output=True)

        if res.fun > max_cost:
            return max_cost
        return res.fun

    def adjust(self):
        res = optimize.brute(
            self.sigmoid_cost_at_xy,
            (
                (self.P0[0] - self.P0_margin, self.P0[0] + self.P0_margin),
                (self.P0[1] - self.P0_margin, self.P0[1] + self.P0_margin)
            ),
            args=(self.img_window, self.r_max,),
            Ns=self.P0_margin * 2 + 1
        )
        self.logger.info(res)
        return np.array(res) + self.offset
        # return np.array(res) + self.offset  # for scipy.optimize.brute


class RSA_Projection_Logic(object):
    """
    Logic related to an RSA image and the projection of an image.
    """

    def __init__(self, dataset, key_string, **kwargs):
        """
        Instantiate rsa image projection from dataset by providing the assessment and image names.

        Parameters
        ----------
        dataset: RSA_Dataset
            Data
        key_string: string
            Root of image data in dataset
        parent: RSA_Assessment_Logic
            Parent assessment of the current image
        """

        assert len(key_string.split(".")) == 4, "Provided key string has incorrect number of levels"

        self.dataset = dataset
        self.key_string = key_string
        self._idict = dataset.get(key_string)
        self._dicom = None
        self._t = None
        self.emitter_position = None
        self.parent = kwargs.get("parent", None)
        self.logger = logging.getLogger(__name__)

    @property
    def t(self):
        """
        Access to this projection's transformation object, instantiated on first access as it requires dicom headers loaded

        Returns
        -------
        instance of app.rsa.transformation.Transformation
        """
        if self._t is None:
            shape_xy = self.dicom.pixel_array.shape[::-1]  # convert to col-major (x,y)

            # TODO: figure out how to handle resolution, might be different for x and y axes
            # the resolution seem to be a little tough to figure out
            # resolution = (np.array(list(self.dicom.ImagerPixelSpacing)) / self.dicom.DistanceSourceToDetector * self.dicom.DistanceSourceToIsocenter).tolist()
            # this shows systematic errors across both x and y movement, two groups of outliers are present both above and below the trend line, not usable

            # also gives systematic outliers both above and below trend line in x and y axis movements
            # resolution = (np.array(list(self.dicom.ImagerPixelSpacing)) / self.dicom.DistanceSourceToDetector * self.dicom.DistanceSourceToPatient).tolist()

            # not tested
            # resolution = list(self.dicom.ImagerPixelSpacing)

            # resolution = [
            #   float(self.dicom.ImagerPixelSpacing[1]),   # convert to col-major to immitate (x,y)
            #   float(self.dicom.ImagerPixelSpacing[0]),
            #   1
            # ]

            # tested, default choice
            resolution = [
                float(self.dicom.PixelSpacing[1]),
                # convert to col-major (x,y) # TODO: Add option to change to ImagerPixelSpacing for nice scene projection
                float(self.dicom.PixelSpacing[0]),
                1
            ]

            self._t = Transformation.factory().scale(resolution)

            if self._idict["name"] == "frontal":
                assert self.dicom.ImageComments == "EOS Frontal"
                self._t.rotate(
                    np.radians([-90, 0, -90])
                ).translate(
                    [
                        0,
                        # TODO: Add option to change to self.dicom.DistanceSourceToDetector - self.dicom.DistanceSourceToIsocenter for nice scene projection
                        shape_xy[0] * resolution[0] / 2,
                        shape_xy[1] * resolution[1]
                    ]
                )
                self.emitter_position = np.array([-self.dicom.DistanceSourceToIsocenter, 0])
            else:  # lateral image
                assert self.dicom.ImageComments == "EOS Lateral"
                self._t.rotate(
                    np.radians([-90, 0, 0])
                ).translate(
                    [
                        -shape_xy[0] * resolution[0] / 2,
                        0,
                        # TODO: Add option to change to self.dicom.DistanceSourceToDetector - self.dicom.DistanceSourceToIsocenter for nice scene projection
                        shape_xy[1] * resolution[1]
                    ]
                )
                self.emitter_position = np.array([0, -self.dicom.DistanceSourceToIsocenter])
        return self._t

    @property
    def dicom(self):
        """
        Access to pydicom dataset object of the dicom file. Loaded on first access.

        Returns
        -------
        pydicom dataset
        """
        if not isinstance(self._dicom, dcm.FileDataset):
            self._dicom = self.dataset.load_dicom(self.key_string + ".dicom_path")
        return self._dicom

    @property
    def assessment_name(self):
        if self.parent is None:
            return None
        return self.parent.name

    @property
    def name(self):
        return self._idict["name"]

    def points_to_lines(self, pp):
        """
        Generate projection lines for given image points

        Parameters
        ----------
        pp: list of (x,y) points

        Returns
        -------
        List of Line objects representing lines from emitter to detected point on the image.
        """
        if len(pp) == 0:
            self.logger.debug("converting empty point list to lines, returning empty list")
            return []

        l = []
        C = np.array(pp)
        C = np.append(C, np.zeros([C.shape[0], 1]), axis=1)  # add z
        C = np.append(C, np.ones([C.shape[0], 1]), axis=1)  # add w
        Ct = self.t.transform(C)

        for i in range(Ct.shape[0]):
            line = Line(
                np.array([
                    self.emitter_position[0],
                    self.emitter_position[1],
                    Ct[i, 2],
                    1,
                ]),
                Ct[i, :]
            )
            l.append(line)

        return l

    def segment_point_lines(self, segment):
        return self.points_to_lines(segment.points)

    def image_segment_list(self):
        """
        Get a list of segment names in this image.

        Returns
        -------
        list of segment names
        """

        return [s["name"] for s in self.dataset.get("segments")]

    def image_segment(self, name):
        assert name in self.image_segment_list(), "Segment '{}' not in dataset under {}".format(name, self.key_string)

        # add list of image segments if it is missing
        if not "image_segments" in self._idict.keys():
            self._idict["image_segments"] = []

        return RSA_Image_Segment.factory(
            self.dataset,
            "{}.image_segments.[name={}]".format(self.key_string, name),
            parent=self
        )

    @property
    def image(self):
        return self.dicom.pixel_array


class RSA_Image_Segment(object):
    TYPE_BEADS = "beads"
    TYPE_SPHERE_SEGMENT = "sphere-segment"

    @classmethod
    def factory(cls, dataset, key_string, **kwargs):
        try:
            name = dataset.get(key_string)["name"]

        # check if segment exist under image segments in dataset, create if not
        except KeyError as e:
            keys = key_string.split(".")
            last_key = keys[-1]
            name = last_key[last_key.find("=") + 1:-1]

            # make sure we are allowed to create this segment
            assert name in kwargs.get("parent").image_segment_list()

            image_segments = dataset.get(".".join(keys[:-1]))
            image_segments.append({
                "name": name,
                "points": []
            })

        _type = dataset.get("segments.[name={}]".format(name))["type"]

        if _type == cls.TYPE_BEADS:
            return RSA_Image_Bead_Segment(dataset, key_string, **kwargs)

        raise NotImplementedError

    def __init__(self, dataset, key_string, **kwargs):
        """
        Instantiate an RSA image segment of beads.
        Parameters
        ----------
        dataset: RSA_Dataset
            Dataset to make instance from
        key_string: string
            Data root in dataset
        """

        self.dataset = dataset
        self.key_string = key_string
        self.parent = kwargs.get("parent", None)

        self._idict = self.dataset.get(key_string)

    @property
    def name(self):
        """
        Name of this segment. This property is present for backwards compatibility.

        Returns
        -------
        Name as string.
        """
        k = self.key_string.split(".")[-1]
        m = re.match(r'\[(\w+)=([\w /]+)\]', k)
        return m.group(2)

    @property
    def points(self):
        """
        Points in this segments. Returns a list of points. Each point is itself a list of [x,y].

        Please see header of this source file for an explanation of the importance of points.

        Returns
        -------
        List of [x,y] points.
        """
        return self._idict["points"]

    @property
    def _type(self):
        """
        Get the type of image segment, e.g. beads, sphere-segment, model, or other.

        Returns
        -------
        Type of image segment as a string
        """
        return self.dataset.get("segments.[name={}].type".format(self.name))


class RSA_Image_Bead_Segment(RSA_Image_Segment):
    pass


class Line(object):
    # # tolerance for termination of search for shortest line between this and another line, set to 1/1000th of suggested limit for MSE for clinical research
    # tol = 0.00035

    def __init__(self, a, b):
        """
        Initialise a line with vectors from origo to points a and b on the line. a and b are n-dimensional vectors in projective geometry (e.g. x,y,w or x,y,z,w)
        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        self.a = a
        self.b = b

    def r(self, lmbd):
        """
        Return vector r to a point on the line described by r = a + λ(b-a)
        """
        _r = self.a + lmbd * (self.b - self.a)
        _r[-1] = 1
        return _r

    @property
    def length(self):
        return np.linalg.norm(self.b[:-1] - self.a[:-1])

    def __len__(self):
        return self.length()

    # def min_dist_line(self, other):
    #     """
    #     Find the minimum distance to the other line and returns a line representing this. The returned line is from this to other.
    #     """
    #     res = minimize(
    #         lambda lambdas, other: np.linalg.norm(other.r(lambdas[1])[:-1] - self.r(lambdas[0])[:-1]),
    #         (0, 0),
    #         args=(other),
    #         tol=self.tol
    #     )
    #     return Line(self.r(res.x[0]), other.r(res.x[1]))

    def __repr__(self):
        return "Line from {} to {}".format(self.a, self.b)
