import os
import shutil
import tempfile
from datetime import datetime
import unicodedata

import cv2 as cv
import numpy as np

from AnyQt.QtCore import Qt, QTimer, QSize
from AnyQt.QtWidgets import QLabel, QPushButton, QSizePolicy
from AnyQt.QtGui import QImage, QPixmap, QImageReader

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets import gui, widget, settings


class OWCamera(widget.OWWidget):
    name = "Camera"
    # icon = "icons/WebcamCapture.svg"

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        self.vc = cv.VideoCapture(0)
        # vc.set(5, 30)  #set FPS
        self.vc.set(3, 640)  # set width
        self.vc.set(4, 480)  # set height

        self.image_label = QLabel()
        # self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.controlArea.layout().addWidget(self.image_label)
        # self.resize(pixmap.width(), pixmap.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        rval, frame = self.vc.read()
        image = QImage(frame[:, :, ::-1].copy(), frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio | Qt.FastTransformation)
        self.image_label.setPixmap(pix)



if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWCamera).run()
