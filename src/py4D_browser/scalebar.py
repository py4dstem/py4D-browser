from pyqtgraph import functions as fn
from pyqtgraph import getConfigOption
from pyqtgraph import Point
from PyQt5 import QtCore, QtWidgets
from pyqtgraph import GraphicsObject
from pyqtgraph import GraphicsWidgetAnchor
from pyqtgraph import TextItem
import numpy as np
from sigfig import round

__all__ = ["ScaleBar"]


class ScaleBar(GraphicsWidgetAnchor, GraphicsObject):
    """
    Displays a rectangular bar to indicate the relative scale of objects on the view.
    """

    def __init__(
        self,
        pixel_size: float,
        units: str,
        target_relaive_size=0.2,
        width=5,
        brush=None,
        pen=None,
        offset=None,
        nice_numbers=[1, 2, 5, 10],
    ):
        GraphicsObject.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        if brush is None:
            brush = getConfigOption("foreground")
        self.brush = fn.mkBrush(brush)
        self.pen = fn.mkPen(pen)
        self._width = width
        self._target_relative_size = target_relaive_size
        self._nice_numbers = np.array(nice_numbers)

        self.pixel_size = pixel_size
        self.units = units

        if offset is None:
            offset = (0, 0)
        self.offset = offset

        self.bar = QtWidgets.QGraphicsRectItem()
        self.bar.setPen(self.pen)
        self.bar.setBrush(self.brush)
        self.bar.setParentItem(self)

        self.text = TextItem(text="smol", anchor=(0.5, 1))
        self.text.setParentItem(self)

    def changeParent(self):
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self.updateBar)
        self.updateBar()

    def updateBar(self):
        view = self.parentItem()

        if view is None:
            return

        view_width = view.viewRect().width() * self.pixel_size
        target_size = view_width * self._target_relative_size

        exponent = np.floor(np.log10(target_size))
        mantissa = target_size / np.power(10, exponent)

        # Get the "nice" size of the scalebar
        nice_size = (
            self._nice_numbers[np.argmin(np.abs(mantissa - self._nice_numbers))]
            * 10**exponent
        )

        p1 = view.mapFromViewToItem(self, QtCore.QPointF(0, 0))
        p2 = view.mapFromViewToItem(
            self, QtCore.QPointF(nice_size / self.pixel_size, 0)
        )
        w = (p2 - p1).x()
        self.bar.setRect(QtCore.QRectF(-w, 0, w, self._width))
        self.text.setPos(-w / 2.0, 0)
        self.text.setText(f"{round(nice_size,sigfigs=1,output_type=str)} {self.units}")

    def boundingRect(self):
        return QtCore.QRectF()

    def setParentItem(self, p):
        ret = GraphicsObject.setParentItem(self, p)
        if self.offset is not None:
            offset = Point(self.offset)
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret
