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
import random


class Ball:

    def __init__(self, surface,  color: tuple = None, radius=20):
        self.x = random.randint(0, 400)
        self.y = random.randint(0, 400)
        self.radius = radius
        self.speed = [5, 5]
        self.color = color if color is not None else (255, 255, 255)
        self.radius = radius

        self.surface = surface
        self.width = surface.get_width()
        self.height = surface.get_height()

        self._ball = self.draw()

    @property
    def is_active(self):
        return True if self.color != (255, 255, 255) else False

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

        self.surface = pygame.Surface(self.size)

        self.blue = self.create_new_ball()
        self.green = self.create_new_ball()

    def create_new_ball(self):
        return Ball(self.surface)

    @property
    def balls(self):
        return [self.blue, self.green]

    @property
    def active_balls(self):
        return [ball for ball in self.balls if ball.is_active]

    def loop(self):
        """ Main loop of the application """
        self.surface.fill((0, 0, 0))
        for ball in self.active_balls:
            ball.move()
            ball.draw()


class OWBouncingBalls(OWWidget, ConcurrentWidgetMixin):
    name = "BouncingBalls"
    icon = "icons/mywidget.svg"

    want_main_area = False

    class Inputs:
        bb = Input('BouncingBalls', list)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.frame = None

        self.animation = BouncingBalls()
        self.timer = QTimer()
        self.timer.timeout.connect(self.pygame_loop)
        self.timer.start(10)

        self.image = None
        self.current_colors = None

    @Inputs.bb
    def on_input(self, colors):
        self.current_colors = colors
        (blue, blue_rad), (green, green_rad) = colors

        if blue:
            self.animation.blue.color = blue
            self.animation.blue.radius = int(100 * blue_rad)
        else:
            self.animation.blue = self.animation.create_new_ball()

        if green:
            self.animation.green.color = green
            self.animation.green.radius = int(100 * green_rad)
        else:
            self.animation.green = self.animation.create_new_ball()

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
    WidgetPreview(OWBouncingBalls).run()

