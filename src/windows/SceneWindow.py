# -*- coding: utf-8 -*-

"""
TODO: Lav filformatet om så det hedder "projection_1" i stedet for "plane_1" og lav navne om til "plane_rotation", "plane_translation", "tube_origin", "image_path" e.g.
"""

import logging

import cv2
import numpy as np
import visvis as vv


# logger = logging.getLogger("app.windows.SceneWindow")


# lines below might improve rendering in visvis on high-resolution displays like the mac retina, but this is not the case.
# import PyQt5
# from PyQt5 import QtCore

# if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
#    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

# if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
#    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class SceneWindow(object):
    lineColor = (1, 0.757, 0.255)
    lineSize = 0.4
    pointColor = (0.694, 0.129, 0.114)
    pointSize = 2.5
    # see https://github.com/almarklein/visvis/wiki/functions#solidSphere
    lineN = 6
    sphereN = 6
    sphereM = 6

    def __init__(self, assessment_logic):
        self.rsa_assessment = assessment_logic
        self.logger = logging.getLogger(__name__)

    def plot_image_data(self, image_data, transformator, sampling=1.0):
        if not sampling == 1.0:
            self.logger.info("Sampling image by {}".format(sampling))
            image_data = cv2.resize(image_data, dsize=(0, 0), fx=sampling, fy=sampling)

        # list all (x,y) coordinates of any pixel in the image, apply resolution scaling
        x = np.arange(image_data.shape[1]).astype(np.float32)
        y = np.arange(image_data.shape[0]).astype(np.float32)

        if not sampling == 1.0:
            self.logger.info(
                "Multiplying image coordinates by {} according to sampling by {}".format(1 / sampling, sampling))
            x /= sampling
            y /= sampling

        # make meshgrid of (x,y)
        X, Y = np.meshgrid(x, y)

        # build coordinate list
        C = np.array([X.flatten(), Y.flatten()]).T
        C = np.append(C, np.zeros([C.shape[0], 1]), axis=1)  # add z
        C = np.append(C, np.ones([C.shape[0], 1]), axis=1)  # add w

        # project into scene
        Ct = transformator.transform(C)

        # reverse meshgrid
        Xt = Ct[..., 0].reshape(image_data.shape)
        Yt = Ct[..., 1].reshape(image_data.shape)
        Zt = Ct[..., 2].reshape(image_data.shape)

        # plot image
        vv.surf(Xt, Yt, Zt, image_data, aa=3)

    def show(self):
        for projection_name in self.rsa_assessment.projection_list():
            rsa_projection = self.rsa_assessment.projection(projection_name)

            # projection = self.assessment.projection(i)
            # projector = self.projectors[i]
            self.plot_image_data(
                rsa_projection.image,
                rsa_projection.t,
                sampling=0.1
            )

            for image_segment_name in rsa_projection.image_segment_list():
                try:
                    rsa_image_segment = rsa_projection.image_segment(image_segment_name)
                except NotImplementedError:
                    continue

                for line in rsa_projection.segment_point_lines(rsa_image_segment):
                    ps = vv.Pointset(3)
                    ps.append(*list(line.a[:3]))
                    ps.append(*list(line.b[:3]))
                    vl = vv.solidLine(ps, radius=self.lineSize, N=self.lineN)
                    vl.faceColor = self.lineColor

        for scene_segment_name in self.rsa_assessment.scene_segment_list():
            try:
                scene_segment = self.rsa_assessment.scene_segment(scene_segment_name)
            except NotImplementedError:
                continue

            scene_segment.match_crossing_lines()
            for i in range(len(scene_segment.points)):
                vs = vv.solidSphere(
                    translation=tuple(scene_segment.points[i][:3]),
                    scaling=([self.pointSize] * 3),
                    N=self.sphereN,
                    M=self.sphereM
                )
                vs.faceColor = self.pointColor

        # Enter main loop
        app = vv.use()
        app.Run()
