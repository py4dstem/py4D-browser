import pyqtgraph as pg


def pg_point_roi(view_box):
    """
    Point selection.  Based in pyqtgraph, and returns a pyqtgraph CircleROI object.
    This object has a sigRegionChanged.connect() signal method to connect to other functions.
    """
    circ_roi = pg.CircleROI((-0.5, -0.5), (2, 2), movable=True, pen=(0, 9))
    h = circ_roi.addTranslateHandle((0.5, 0.5))
    h.pen = pg.mkPen("r")
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi
