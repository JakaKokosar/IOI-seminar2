import cv2 as cv
import time
import numpy as np

from AnyQt.QtCore import Qt, QTimer, QSize
from AnyQt.QtWidgets import QLabel, QPushButton, QSizePolicy
from AnyQt.QtGui import QImage, QPixmap, QImageReader

from Orange.widgets.widget import OWWidget, Output

from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from dominantcolors import find_dominant_colors


class OWCamera(OWWidget, ConcurrentWidgetMixin):
    name = "Camera"
    icon = "icons/WebcamCapture.svg"

    want_main_area = False

    class Outputs:
        frame = Output("Frame", list)
        colors = Output("Colors", list)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        # self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        self.vc = cv.VideoCapture(0)
        self.vc.set(3, 640)  # set width
        self.vc.set(4, 480)  # set height

        self.image_label = QLabel()
        # self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.controlArea.layout().addWidget(self.image_label)
        # self.resize(pixmap.width(), pixmap.height())
        #
        # self.frame_timer = QTimer()
        # self.frame_timer.timeout.connect(self.update_frame)
        # self.frame_timer.start(0)
        #
        # self.output_timer = QTimer()
        # self.output_timer.timeout.connect(self.commit)
        # self.output_timer.start(0)

        # self.current_frame = None
        self.start(self.worker)
        self.setBlocking(False)

    def worker(self, state: TaskState):
        while True:
            state.set_partial_result(self.update_frame())
            time.sleep(1/10)

    def on_partial_result(self, result):
        frame, colors = result
        self.Outputs.frame.send(frame)
        self.Outputs.colors.send(colors)

    def update_frame(self):
        # read pixle values from camera
        rval, frame = self.vc.read()
        #
        if frame.size:
            # display image in orange widget
            image = QImage(frame[:, :, ::-1].copy(), frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio | Qt.FastTransformation)
            self.image_label.setPixmap(pix)

        # transform to 2D matrix
        # self.current_frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
        frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
        return frame, find_dominant_colors(frame, 3)

    def onDeleteWidget(self):
        self.vc.release()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWCamera).run()

