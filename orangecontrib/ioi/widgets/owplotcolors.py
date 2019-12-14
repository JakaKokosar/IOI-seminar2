import os
import shutil
import tempfile
from datetime import datetime
import unicodedata
#
import cv2 as cv
import numpy as np
#
from AnyQt.QtCore import Qt, QTimer, QSize
from AnyQt.QtWidgets import QLabel, QPushButton, QSizePolicy
from AnyQt.QtGui import QImage, QPixmap, QImageReader
#
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import OWWidget, Input, Output
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt

from dominantcolors import find_dominant_colors

from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin


from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.figure import Figure

if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


class OWPlotColors(OWWidget, ConcurrentWidgetMixin):
    name = "Plot colors"
    icon = "icons/mywidget.svg"

    want_main_area = False

    class Inputs:
        frame = Input('Frame', list)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._dynamic_ax.axis('off')
        self.controlArea.layout().addWidget(dynamic_canvas)

        self.button = QPushButton('Determine dominant colors using K-means', self)
        self.button.clicked.connect(self.on_click)
        self.controlArea.layout().addWidget(self.button)

        self.k_means = KMeans(n_clusters=3)
        self.frame = None

    def on_click(self):
        if self.frame is None:
            return

        self.start(self.k_means_fit, self.frame)

    def k_means_fit(self, frame, state: TaskState):
        self.k_means.fit(frame)

    def on_done(self, result):
        height = 100
        width = 200

        hist, _ = np.histogram(self.k_means.labels_, bins=np.arange(0, len(np.unique(self.k_means.labels_)) + 1))
        hist = hist.astype('float')
        hist /= hist.sum()

        image = np.zeros((height, width, 3), dtype='uint8')
        start = 0
        for percent, (R, G, B) in zip(hist, self.k_means.cluster_centers_):
            # plot the relative percentage of each cluster
            end = start + (percent * width)
            cv.rectangle(image, (int(start), 0), (int(end), height), (R, G, B), -1)
            start = end

        self._dynamic_ax.clear()
        self._dynamic_ax.axis('off')
        self._dynamic_ax.imshow(image)
        self._dynamic_ax.figure.canvas.draw()

    @Inputs.frame
    def on_input(self, frame):
        self.frame = frame

    def onDeleteWidget(self):
        super().onDeleteWidget()

    def __set_state_ready(self):
        self.setBlocking(False)

    def __set_state_busy(self):
        self.setBlocking(True)

    def _connect_signals(self, state: TaskState):
        super()._connect_signals(state)

    def _disconnect_signals(self, state: TaskState):
        super()._disconnect_signals(state)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPlotColors).run()

