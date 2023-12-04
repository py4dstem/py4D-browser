import pyqtgraph as pg
import numpy as np


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


def make_detector(shape: tuple, mode: str, geometry) -> np.ndarray:
    match mode, geometry:
        case ["point", (qx, qy)]:
            mask = np.zeros(shape, dtype=np.bool_)
            mask[qx, qy] = True
        case ["point", geom]:
            raise ValueError(
                f"Point detector shape must be specified as (qx,qy), not {geom}"
            )

        case [("circle" | "circular"), ((qx, qy), r)]:
            ix, iy = np.indices(shape)
            mask = np.hypot(ix - qx, iy - qy) <= r
        case [("circle" | "circular"), geom]:
            raise ValueError(
                f"Circular detector shape must be specified as ((qx,qy),r), not {geom}"
            )

        case [("annulus" | "annular"), ((qx, qy), (ri, ro))]:
            ix, iy = np.indices(shape)
            ir = np.hypot(ix - qx, iy - qy)
            mask = np.logical_and(ir >= ri, ir <= ro)
        case [("annulus" | "annular"), geom]:
            raise ValueError(
                f"Annular detector shape must be specified as ((qx,qy),(ri,ro)), not {geom}"
            )

        case [("rectangle" | "square" | "rectangular"), (xmin, xmax, ymin, ymax)]:
            mask = np.zeros(shape, dtype=np.bool_)
            mask[xmin:xmax, ymin:ymax] = True
        case [("rectangle" | "square" | "rectangular"), geom]:
            raise ValueError(
                f"Rectangular detector shape must be specified as (xmin,xmax,ymin,ymax), not {geom}"
            )

        case ["mask", mask_arr]:
            mask = mask_arr

        case unknown:
            raise ValueError(f"mode and geometry not understood: {unknown}")

    return mask
