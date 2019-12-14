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


class Ball:

    def __init__(self, surface, x, y, color: tuple, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = [1, 1]
        self.color = color

        self.surface = surface
        self.width = surface.get_width()
        self.height = surface.get_height()

        self._ball = self.draw()

    def draw(self):
        return pygame.draw.circle(self.surface, self.color, (self.x, self.y), self.radius)

    def move(self):
        self._ball = self._ball.move(self.speed)
        self.x, self.y = self._ball.center

        if self._ball.left < 0 or self._ball.right > self.width:
            self.speed[0] = -self.speed[0]
        if self._ball.top < 0 or self._ball.bottom > self.height:
            self.speed[1] = -self.speed[1]


class BouncingBalls:

    def __init__(self):
        # pygame.init()

        self.size = self.width, self.height = 680, 480
        self.speed = [1, 1]
        self.white = 255, 255, 255
        self.black = 0, 0, 0

        self.surface = pygame.Surface(self.size)

        self._balls = [Ball(self.surface, 0, 0, self.white, 30),
                       Ball(self.surface, 10, 10, self.white, 30),
                       Ball(self.surface, 20, 20, self.white, 30)]

        # image = np.zeros((20, 20, 3), dtype='uint8')
        # image = cv.circle(image, (10, 10), 10, (255, 255, 255), -1)
        # self.ball = pygame.image.frombuffer(image, image.shape[1::-1], "RGB")
        # self.ball = pygame.draw.circle(self.surface, self.white, (0, 0), 10)

    def loop(self):
        """ Main loop of the application """
        self.surface.fill(self.black)
        for ball in self._balls:
            ball.move()
            ball.draw()
        # self.ball = pygame.draw.circle(self.surface, self.white, self.ball.center, 20)
        self.surface.blit(self.surface, (0, 0))


class OWAnimation(OWWidget, ConcurrentWidgetMixin):
    name = "Animation"
    icon = "icons/mywidget.svg"

    want_main_area = False

    class Inputs:
        colors = Input('Colors', list)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.frame = None

        self.animation = BouncingBalls()
        self.timer = QTimer()
        self.timer.timeout.connect(self.pygame_loop)
        self.timer.start(0)

        self.image = None
        self.current_colors = None

        # self.image = None
        # pygame.init()
        # # surface = pygame.display.set_mode((640, 480))
        # # surface = pygame.Surface((640, 480))
        # surface.fill((64, 128, 192, 224))
        # pygame.draw.circle(surface, (255, 255, 255, 255), (100, 100), 50)
        #
        # w = surface.get_width()
        # h = surface.get_height()
        # self.data = surface.get_buffer().raw
        # self.image = QImage(self.data, w, h, QImage.Format_RGB32)

    @Inputs.colors
    def on_input(self, colors):
        self.current_colors = colors

        c1, c2, c3 = colors
        self.animation._balls[0].color = tuple(c1)
        self.animation._balls[1].color = tuple(c2)
        self.animation._balls[2].color = tuple(c3)

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
    WidgetPreview(OWAnimation).run()

