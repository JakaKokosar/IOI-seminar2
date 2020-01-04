import cv2 as cv
import numpy as np

from Orange.widgets.widget import OWWidget, Input
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
        DominantColors = Input('DominantColors', list)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._dynamic_ax.axis('off')
        self.controlArea.layout().addWidget(dynamic_canvas)

    @Inputs.DominantColors
    def on_input(self, dominant_colors):
        centroids, hist = dominant_colors

        height = 100
        width = 200
        #
        # hist, _ = np.histogram(labels, bins=np.arange(0, len(np.unique(labels)) + 1))
        # hist = hist.astype('float')
        # hist /= hist.sum()

        image = np.zeros((height, width, 3), dtype='uint8')
        start = 0

        for percent, (R, G, B) in sorted(zip(hist, centroids), key=lambda x: x[0], reverse=True):
            # plot the relative percentage of each cluster
            end = start + (percent * width)
            cv.rectangle(image, (int(start), 0), (int(end), height), (int(R), int(G), int(B)), -1)
            start = end

        self._dynamic_ax.clear()
        self._dynamic_ax.axis('off')
        self._dynamic_ax.imshow(image)
        self._dynamic_ax.figure.canvas.draw()

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

