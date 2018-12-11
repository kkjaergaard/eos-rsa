"""

UI for displaying an RSA image and modifying relevant data on the image

TODO: modify statusbar to accommodate name of file, assessment and image, separate field with (x,y, value), and loading bar
TODO: add log window, hook up to logging, add options to clear, copy, and save to file, see https://stackoverflow.com/a/28794076

"""

import copy
import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from misc import Signal
from rsa.projection import RSA_Image_Segment, BeadCenterAdjuster


class SegmentCtrl(object):
    """
    Segment controller, handles UI events and backpropagation into segment data structure
    """

    def __init__(self, imageView, segment, active=False):
        """
        :param imageView: instance of ImageView, should have member "scene" with mouse signals
        :param segment: segment class
        :param active: whether this segment is active at instantiation
        """
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.imageView = imageView
        self.segment = segment
        self.active = active
        if active:
            self.activate()

        self.sigChanged = Signal()

    def is_active(self):
        """
        Returns true or false whether the state of this controller is active or not
        :return:
        """
        return self.active


class BeadsSegmentCtrl(SegmentCtrl):
    """
    Segment controller for simple points
    """

    def __init__(self, imageView, segment, active=False):
        super().__init__(imageView, segment, active)
        self.roi_list = []
        self.segment = segment

        self.pg_roi_offset = 0.0
        if pg.__version__ == "0.10.0":
            # a crosshair roi at (0,0) points to the top-left corner, not the middle of the pixel
            # this offset is added when crosshairs are displayed, and subtracted when saved to the dataset
            self.pg_roi_offset = 0.5

        # add ROIs from segment to the view,
        for (x, y) in self.segment.points:
            roi = pg.CrosshairROI((x + self.pg_roi_offset, y + self.pg_roi_offset), (20, 20), removable=True)
            roi.sigRemoveRequested.connect(self.removeRoiEvent)
            roi.sigRegionChanged.connect(self.translateRoiEvent)
            roi.setVisible(False)
            self.roi_list.append(roi)
            self.imageView.view.addItem(roi)

        self.pf_canvas = np.array([15, 15])

    def activate(self):
        """
        Make segment ROIs visible and connect slots to scene
        :return:
        """
        if not self.active:
            # register click event on scene
            self.imageView.scene.sigMouseClicked.connect(self.sceneClickEvent)

            # display all ROIs in this segment
            for roi in self.roi_list:
                roi.setVisible(True)
        self.active = True
        self.logger.debug("Registered segment {} for ui events".format(self.segment.name))

    def deactivate(self):
        """
        Make segment ROIs invisible and disconnect slots to scene
        :return:
        """
        if self.active:
            # disconnect from mouse click signals
            self.imageView.scene.sigMouseClicked.disconnect(self.sceneClickEvent)

            # hide all ROIs in this segment
            for roi in self.roi_list:
                roi.setVisible(False)
        self.active = False
        self.logger.debug("Unregistered segment {} for ui events".format(self.segment.name))

    def removeRoiEvent(self, roi):
        """
        Remove an ROI from the view, the list of ROIs and the segment data structure
        :param roi:
        :return:
        """
        self.imageView.view.removeItem(roi)
        idx = self.roi_list.index(roi)
        del self.roi_list[idx]
        del self.segment.points[idx]
        self.logger.debug(
            "Deleted ROI at ({:.2f},{:.2f}), now {} has {} ROIs in list and {} in segment list".format(*list(roi.pos()),
                self.segment.name, len(self.roi_list), len(self.segment.points)))

        self.sigChanged.emit()

    def translateRoiEvent(self, roi):
        idx = self.roi_list.index(roi)
        x, y = copy.deepcopy(roi.pos())
        self.segment.points[idx] = [x - self.pg_roi_offset, y - self.pg_roi_offset]
        self.logger.debug("Moved ROI {} to ({:.2f},{:.2f})".format(idx, x, y))
        self.sigChanged.emit()

    def sceneClickEvent(self, event):
        """
        Scene click slot for adding new ROIs
        :param event:
        :return:
        """

        self.logger.debug("Received scene click signal at segment: {}".format(self.segment.name))

        modifiers = event.modifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            # TODO: Add keyboard modifier: shift+click should save the window to a file
            self.logger.warning("shift click save to file not yet implemented")

        if event.buttons() & QtCore.Qt.LeftButton:
            # scene is clicked with left mouse button, then we add an ROI to the scene, this class' list, and the segment point list
            P = self.imageView.view.mapSceneToView(event.scenePos())
            x, y = (int(P.x()), int(P.y()))

            center_adjuster = BeadCenterAdjuster.factory(self.imageView.image, x, y, BeadCenterAdjuster.METHOD_SIGMOID)
            try:
                x_adj, y_adj = [float(x) for x in center_adjuster.adjust()]
            except Exception as e:
                self.logger.warning("An error occurred during center adjustment: {}".format(e))
                x_adj = x
                y_adj = y

            roi = pg.CrosshairROI((x_adj + self.pg_roi_offset, y_adj + self.pg_roi_offset), (20, 20), removable=True)
            roi.setVisible(self.active)
            roi.sigRemoveRequested.connect(self.removeRoiEvent)
            self.imageView.view.addItem(roi)
            self.roi_list.append(roi)
            self.segment.points.append([x_adj, y_adj])
            self.logger.debug(
                "Added ROI at ({:.2f},{:.2f}), now {} has {} ROIs in list and {} in segment list".format(
                    x_adj,
                    y_adj,
                    self.segment.name,
                    len(self.roi_list),
                    len(self.segment.points)
                )
            )
            self.sigChanged.emit()

            event.accept()
            return

        event.ignore()


class ImageWindow(QtGui.QMainWindow):
    # limit GUI refresh rate
    refresh_rate = 30

    def __init__(self, rsa_series_logic):
        super().__init__(None)
        self.logger = logging.getLogger(__name__)

        pg.setConfigOptions(
            # set pg to row-major like numpy as opencv
            imageAxisOrder="row-major"
        )

        uic.loadUi(Path(__file__).resolve().parent.joinpath("ImageWindow.ui"), self)

        # load and adjust UI
        # self.ui = Ui_PlaneWindow()
        # self.setupUi(self)
        self.imageView.view.setBackgroundColor("#666666")

        self.series = rsa_series_logic
        self.unsaved_changes = False
        self.image = None

        # setup image combobox
        for assessment_name in self.series.assessment_list():
            assessment = self.series.assessment(assessment_name)

            for projection_name in assessment.projection_list():
                projection = assessment.projection(projection_name)

                self.projectionComboBox.addItem(
                    "{}/{}".format(assessment_name, projection_name),
                    userData=projection
                )
        # register event slot for projection change and trigger event to set up rest of UI
        self.projectionComboBox.currentIndexChanged.connect(self.changeProjectionEvent)
        self.changeProjectionEvent(self.projectionComboBox.currentIndex())

        # keyboard shortcuts
        # self.cycleAssessmentShortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+A"), self)

        # slots
        self.sceneMouseMovedProxy = pg.SignalProxy(
            self.imageView.scene.sigMouseMoved,
            rateLimit=self.refresh_rate,
            slot=self.sceneMouseMovedEvent)  # used to limit update frequency

        # segment selection events
        self.segmentComboBox.currentIndexChanged.connect(self.changeSegmentEvent)

    def datasetChanged(self):
        if not self.unsaved_changes:
            pass

        self.unsaved_changes = True

    def changeSegmentEvent(self, new_index):
        # deactivate any active segments
        for i in range(self.segmentComboBox.count()):
            itm = self.segmentComboBox.itemData(i)
            if isinstance(itm, SegmentCtrl):
                if itm.is_active():
                    itm.deactivate()

        # activate selected segment
        currentItm = self.segmentComboBox.currentData()
        if isinstance(currentItm, SegmentCtrl):
            currentItm.activate()

    def changeProjectionEvent(self, new_index):
        # unregister any current handlers for the image
        if self.segmentComboBox.count() > 0:
            # the first item in segmentComboBox is always the (none) item, by activating this all segment handlers are unregistered
            self.segmentComboBox.setCurrentIndex(0)

        # remove any items in segmentComboBox
        idx = list(range(self.segmentComboBox.count()))[::-1]
        for i in idx:
            self.segmentComboBox.removeItem(i)  # this does not trigger the index changed event in the combo box

        new_projection = self.projectionComboBox.currentData()

        self.image = new_projection.image

        # load and display image
        self.imageView.setImage(self.image)
        s = self.image.shape
        self.imageView.view.setLimits(xMin=-int(s[1] / 2.0), yMin=-int(s[0] / 2.0), xMax=int(s[1] * 1.5),
            yMax=int(s[0] * 1.5))
        self.imageView.view.setAspectLocked(True)

        # fill in segments
        self.segmentComboBox.addItem("(none)")
        for segment_name in new_projection.image_segment_list():
            try:
                image_segment = new_projection.image_segment(segment_name)
            except NotImplementedError:
                continue

            segmentCtrl = None
            if image_segment._type == RSA_Image_Segment.TYPE_BEADS:
                segmentCtrl = BeadsSegmentCtrl(self.imageView, image_segment)
                segmentCtrl.sigChanged.connect(self.datasetChanged)

            if segmentCtrl is None:
                self.logger.warning("No segment controller found for segment '{}' of type '{}'".format(segment_name,
                    image_segment._type))
                continue

            self.segmentComboBox.addItem(
                image_segment.name,
                userData=segmentCtrl
            )

    def sceneMouseMovedEvent(self, event):
        """
        Slot to display coordinates and pixel values in the statubar when mouse is moved across image
        :param event:
        :return:
        """
        P = self.imageView.view.mapSceneToView(event[0])
        x, y = (int(P.x()), int(P.y()))
        if x >= 0 and x < self.image.shape[1] and y >= 0 and y < self.image.shape[0]:
            # print useful info to statusbar
            msg = "(x,y) = (%d,%d), value = %d" % (x, y, self.image[y, x])
            self.statusBar.showMessage(msg, 5000)
        else:
            self.statusBar.clearMessage()

    def closeEvent(self, event):
        if self.unsaved_changes:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm",
                "Save changes?",
                QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.logger.debug("saving changes to {}".format(self.series.dataset.fpath))
                self.series.dataset.flush()
