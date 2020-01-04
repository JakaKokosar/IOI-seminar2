import os
import time
import shutil
import tempfile
from datetime import datetime
import unicodedata
#
import cv2 as cv
import numpy as np
#
from itertools import accumulate
from math import sin, cos, radians
from AnyQt.QtCore import Qt, QTimer, QSize

from AnyQt.QtWidgets import QLabel, QPushButton, QSizePolicy
from AnyQt.QtGui import QImage, QPixmap, QImageReader, QPainter
#
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import OWWidget, Input, Output
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt

from dominantcolors import find_dominant_colors

from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
import pygame
import random


class Animation:
    def __init__(self):
        # pygame.init()

        self.size = self.width, self.height = 900, 900
        self.surface = pygame.Surface(self.size)
        self.t = 0

        self.factor = 50
        # self.num_of_lines = 15
        self.num_of_lines = list(range(0, 15))
        self.length_to_split = [1, 2, 3, 4, 5]
        self.line_buckets = [self.num_of_lines[x - y: x]
                             for x, y in zip(accumulate(self.length_to_split), self.length_to_split)]

        self.black = (0, 0, 0)
        self.hist = [0] * len(self.length_to_split)
        self.centroids = [self.black] * len(self.length_to_split)

    def x(self, t):
        t = radians(t)
        return (sin(t / 10) * 300 + sin(t / 5) * 200)

    def y(self, t):
        t = radians(t)
        return (cos(t / 10) * 300 + sin(t / 5) * 100)

    def x2(self, t):
        t = radians(t)
        return (sin(t / 10) * 400 + sin(t) * 2)

    def y2(self, t):
        t = radians(t)
        return (cos(t / 20) * 400 + cos(t / 12) * 20)

    def loop(self):
        """ Main loop of the application """
        self.surface.fill((0, 0, 0))
        sorted_by_hist = sorted(zip(self.hist, self.centroids), key=lambda x: x[0], reverse=False)

        for (_, color), lines in zip(sorted_by_hist, self.line_buckets):
            for i in lines:

                x1 = self.x(self.t + (i * self.factor)) + self.surface.get_width() // 2
                y1 = self.surface.get_height() // 2 - self.y(self.t + (i * self.factor))

                x2 = self.x2(self.t + (i * self.factor)) + self.surface.get_width() // 2
                y2 = self.surface.get_height() // 2 - self.y2(self.t + (i * self.factor))

                pygame.draw.line(self.surface,
                                 color,
                                 (x1, y1), (x2, y2), 5)

        self.t = self.t + 2.5
        # self.surface.blit(self.surface, (self.surface.get_width() // 2,  self.surface.get_height() // 2 ))
        # self.surface.blit(self.surface, (680// 2,  480 // 2 ))


class OWScreenSaver(OWWidget, ConcurrentWidgetMixin):
    name = "Animation"
    icon = "icons/mywidget.svg"

    want_main_area = False

    class Inputs:
        DominantColors = Input('DominantColors', list)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.frame = None

        self.animation = Animation()
        self.timer = QTimer()
        self.timer.timeout.connect(self.pygame_loop)
        self.timer.start(5)

        self.image = None
        self.current_colors = None

    @Inputs.DominantColors
    def on_input(self, dominant_colors):
        rgbs, hist = dominant_colors
        self.animation.centroids = rgbs
        self.animation.hist = hist

    def pygame_loop(self):
        self.animation.loop()

        self.image = QImage(self.animation.surface.get_buffer().raw,
                            self.animation.surface.get_width(),
                            self.animation.surface.get_height(),
                            QImage.Format_RGB32)

        # repaint Qt window
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            qp = QPainter()
            qp.begin(self)
            qp.drawImage(0, 0, self.image)
            qp.end()

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
    WidgetPreview(OWScreenSaver).run()

