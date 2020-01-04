import cv2 as cv
import time
import numpy as np
import imutils

from AnyQt.QtCore import Qt, QTimer, QSize
from AnyQt.QtWidgets import QLabel, QPushButton, QSizePolicy
from AnyQt.QtGui import QImage, QPixmap, QImageReader

from Orange.widgets.widget import OWWidget, Output

from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from dominantcolors import find_dominant_colors

from sklearn.cluster import KMeans
from collections import Counter
import colorsys
from matplotlib.colors import hsv_to_rgb

S_MIN = 100
V_MIN = 100

# # define range of orange color in HSV
# ORANGE_MIN = np.array([10, S_MIN, V_MIN], np.uint8)
# ORANGE_MAX = np.array([20, 255, 255], np.uint8)
#
# # define range of yellow color in HSV
# YELLOW_MIN = np.array([25, S_MIN, V_MIN], np.uint8)
# YELLOW_MAX = np.array([35, 255, 255], np.uint8)
#
# LIGHT_RED_MIN = np.array([0, S_MIN, V_MIN], np.uint8)
# LIGHT_RED_MAX = np.array([5, 255, 255], np.uint8)
#
# # define range of red color in HSV
# DARK_RED_MIN = np.array([160, S_MIN, V_MIN], np.uint8)
# DARK_RED_MAX = np.array([179, 255, 255], np.uint8)

# define range of blue color in HSV
BLUE_MIN = np.array([90, S_MIN, V_MIN])
BLUE_MAX = np.array([130, 255, 255])

# define range of green color in HSV
GREEN_MIN = np.array([40, S_MIN, V_MIN], np.uint8)
GREEN_MAX = np.array([80, 255, 255], np.uint8)

BOUNDARIES = (
    (BLUE_MIN, BLUE_MAX),
    (GREEN_MIN, GREEN_MAX),
)


def find_image_colors(image, k=5, image_processing_size=None):

    if image_processing_size is not None:
        image = cv.resize(image, image_processing_size, interpolation=cv.INTER_AREA)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    clt.fit_predict(image)
    rgbs = [tuple(cv.cvtColor(np.uint8([[hsv]]), cv.COLOR_HSV2RGB).flatten()) for hsv in clt.cluster_centers_]

    hist, _ = np.histogram(clt.labels_, bins=np.arange(0, len(np.unique(clt.labels_)) + 1))
    hist = hist.astype('float')
    hist /= hist.sum()

    return rgbs, hist

    # centroids, labels = clt.cluster_centers_, clt.labels_
    # centroids = np.asanyarray(find_dominant_colors(image, 6))
    # print( centroids)

    # colors = [None, None, None, None]
    # print(type(centroids), centroids)
    # mask = (centroids < ORANGE_MAX) & (centroids > ORANGE_MIN)
    # orange_values = centroids[np.all(mask, axis=1), :]
    # if orange_values.size >= 1:
    #     orange_values = np.mean(orange_values, axis=0)
    #     colors[0] = tuple(cv.cvtColor(np.uint8([[orange_values]]), cv.COLOR_HSV2RGB).flatten())

    # mask = (centroids < YELLOW_MAX) & (centroids > YELLOW_MIN)
    # yellow_values = centroids[np.all(mask, axis=1), :]
    # if yellow_values.size >= 1:
    #     yellow_values = np.mean(yellow_values, axis=0)
    #     colors[0] = tuple(cv.cvtColor(np.uint8([[yellow_values]]), cv.COLOR_HSV2RGB).flatten())
    #
    # mask = (centroids < np.array([10, 255, 255], np.uint8)) & (centroids > np.array([0, 0, 0], np.uint8))
    # red_values = centroids[np.all(mask, axis=1), :]
    # if red_values.size >= 1:
    #     red_values = np.mean(red_values, axis=0)
    #     colors[0] = tuple(cv.cvtColor(np.uint8([[red_values]]), cv.COLOR_HSV2RGB).flatten())
    #
    # mask = (centroids < np.array([179, 255, 255], np.uint8)) & (centroids > np.array([160, 0, 0], np.uint8))
    # red_values = centroids[np.all(mask, axis=1), :]
    # if red_values.size >= 1:
    #     red_values = np.mean(red_values, axis=0)
    #     colors[1] = tuple(cv.cvtColor(np.uint8([[red_values]]), cv.COLOR_HSV2RGB).flatten())
    #
    # mask = (centroids < BLUE_MAX) & (centroids > BLUE_MIN)
    # blue_values = centroids[np.all(mask, axis=1), :]
    # if blue_values.size >= 1:
    #     blue_values = np.mean(blue_values, axis=0)
    #     colors[2] = tuple(cv.cvtColor(np.uint8([[blue_values]]), cv.COLOR_HSV2RGB).flatten())
    #
    # mask = (centroids < GREEN_MAX) & (centroids > GREEN_MIN)
    # green_values = centroids[np.all(mask, axis=1), :]
    # if green_values.size >= 1:
    #     green_values = np.mean(green_values, axis=0)
    #     colors[3] = tuple(cv.cvtColor(np.uint8([[green_values]]), cv.COLOR_HSV2RGB).flatten())
    #
    # print(colors)
    # return colors


class OWCamera(OWWidget, ConcurrentWidgetMixin):
    name = "Camera"
    icon = "icons/WebcamCapture.svg"

    want_main_area = False

    class Outputs:
        DominantColors = Output("DominantColors", list)
        BouncingBalls = Output("BouncingBalls", list)

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
        dominant, bb = result
        self.Outputs.DominantColors.send(dominant)
        self.Outputs.BouncingBalls.send(bb)

    def remap(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def update_frame(self):
        # read pixle values from camera
        _, frame = self.vc.read()
        balls = [None, None]

        if frame.size:
            # display image in orange widget
            image = QImage(frame[:, :, ::-1].copy(), frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio | Qt.FastTransformation)
            self.image_label.setPixmap(pix)

        # Convert BGR to HSV
        hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # accumulated_mask = np.zeros(hsv_image.shape[:2], dtype='uint8')

        for index, (lower, upper) in enumerate(BOUNDARIES):
            mask = cv.inRange(hsv_image, lower, upper)
            size = self.remap(np.count_nonzero(mask), 0, mask.size, 0.2, 1)

            rgb_value = (cv.cvtColor(np.uint8([[cv.mean(hsv_image, mask)[:3]]]), cv.COLOR_HSV2RGB).flatten())
            balls[index] = (tuple(rgb_value) if rgb_value.any() else None, size)

            # accumulated_mask = cv.bitwise_or(accumulated_mask, mask)

        dominant_colors = find_image_colors(hsv_image, image_processing_size=(100, 100))
        # accumulated_mask = cv.bitwise_not(accumulated_mask)
        # res = cv.bitwise_and(frame, frame, mask=accumulated_mask)

        # print(res.reshape((res.shape[0] * res.shape[1], 3)).shape)
        # print(cv.mean(hsv_image, accumulated_mask))

        # res = cv.resize(res, (100, 100), interpolation=cv.INTER_AREA)

        # if res.size:
        #     # display image in orange widget
        #     image = QImage(res[:, :, ::-1].copy(), res.shape[1], res.shape[0], QImage.Format_RGB888)
        #     pix = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio | Qt.FastTransformation)
        #     self.image_label.setPixmap(pix)

        # res = res.reshape((res.shape[0] * res.shape[1], 3))
        # res = res[~(res == 0).all(1)]
        # # # print(res)
        #
        # colors = find_image_colors(res, k=6, image_processing_size=(150, 150))

        return dominant_colors, balls

    def onDeleteWidget(self):
        self.vc.release()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWCamera).run()

