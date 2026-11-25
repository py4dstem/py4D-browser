import numpy as np
import pyqtgraph as pg
from functools import partial
from PyQt5 import QtCore
from py4D_browser.utils import (
    pg_point_roi,
    make_detector,
    DetectorShape,
    DetectorMode,
    DetectorInfo,
    RectangleGeometry,
    CircleGeometry,
    AnnulusGeometry,
    PointGeometry,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


def get_diffraction_detector(self: "DataViewer") -> DetectorInfo:
    """
    Get the current detector and its position on the diffraction view.
    Returns a DetectorInfo dictionary, which contains the shape and
    response mode of the detector and information on the selection
    it represents. The selection is described using one (or more) of
    the `slice`, `mask`, and `point` entries, depending on the detector
    type. The selections are expressed in data coordinates.
    """
    shape = DetectorShape(self.detector_shape_group.checkedAction().text())
    mode = DetectorMode(self.detector_mode_group.checkedAction().text())

    match shape:
        case DetectorShape.POINT:
            roi_state = self.virtual_detector_point.saveState()
            y0, x0 = roi_state["pos"]
            xc, yc = int(x0 + 1), int(y0 + 1)

            # Normalize coordinates
            xc = np.clip(xc, 0, self.datacube.Q_Nx - 1)
            yc = np.clip(yc, 0, self.datacube.Q_Ny - 1)

            return DetectorInfo(
                shape=shape,
                mode=mode,
                point=[xc, yc],
                geometry=PointGeometry(x=xc, y=yc),
            )

        case DetectorShape.RECTANGULAR:
            slices, _ = self.virtual_detector_roi.getArraySlice(
                self.datacube.data[0, 0, :, :].T,
                self.diffraction_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices

            mask = np.zeros(self.datacube.Qshape, dtype=np.bool_)
            mask[slice_x, slice_y] = True

            return DetectorInfo(
                shape=shape,
                mode=mode,
                slice=[slice_x, slice_y],
                mask=mask,
                geometry=RectangleGeometry(
                    xmin=slice_x.start,
                    xmax=slice_x.stop,
                    ymin=slice_y.start,
                    ymax=slice_y.stop,
                ),
            )
        case DetectorShape.CIRCLE:
            R = self.virtual_detector_roi.size()[0] / 2.0

            x0 = self.virtual_detector_roi.pos()[1] + R
            y0 = self.virtual_detector_roi.pos()[0] + R

            mask = make_detector(
                (self.datacube.Q_Nx, self.datacube.Q_Ny), "circle", ((x0, y0), R)
            )

            return DetectorInfo(
                shape=shape,
                mode=mode,
                mask=mask,
                geometry=CircleGeometry(x=x0, y=y0, R=R),
            )

        case DetectorShape.ANNULUS:
            inner_pos = self.virtual_detector_roi_inner.pos()
            inner_size = self.virtual_detector_roi_inner.size()
            R_inner = inner_size[0] / 2.0
            x0 = inner_pos[1] + R_inner
            y0 = inner_pos[0] + R_inner

            outer_size = self.virtual_detector_roi_outer.size()
            R_outer = outer_size[0] / 2.0

            if R_inner <= R_outer:
                R_inner -= 1

            mask = make_detector(
                (self.datacube.Q_Nx, self.datacube.Q_Ny),
                "annulus",
                ((x0, y0), (R_inner, R_outer)),
            )

            return DetectorInfo(
                shape=shape,
                mode=mode,
                mask=mask,
                geometry=AnnulusGeometry(x=x0, y=y0, R_inner=R_inner, R_outer=R_outer),
            )

        case _:
            raise ValueError("Detector could not be determined")


def get_virtual_image_detector(self: "DataViewer") -> DetectorInfo:
    """
    Get the current detector and its position on the diffraction view.
    Returns a DetectorInfo dictionary, which contains the shape and
    response mode of the detector and information on the selection
    it represents. The selection is described using one (or more) of
    the `slice`, `mask`, and `point` entries, depending on the detector
    type. The selections are expressed in data coordinates.
    """
    shape = DetectorShape(self.rs_detector_shape_group.checkedAction().text())
    mode = DetectorMode(self.realspace_detector_mode_group.checkedAction().text())

    match shape:
        case DetectorShape.POINT:
            roi_state = self.real_space_point_selector.saveState()
            y0, x0 = roi_state["pos"]
            xc, yc = int(x0 + 1), int(y0 + 1)

            # Normalize coordinates
            xc = np.clip(xc, 0, self.datacube.R_Nx - 1)
            yc = np.clip(yc, 0, self.datacube.R_Ny - 1)

            return DetectorInfo(
                shape=shape,
                mode=mode,
                point=[xc, yc],
                geometry=PointGeometry(x=xc, y=yc),
            )

        case DetectorShape.RECTANGULAR:
            slices, _ = self.real_space_rect_selector.getArraySlice(
                np.zeros((self.datacube.Rshape)).T,
                self.real_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices

            mask = np.zeros(self.datacube.Rshape, dtype=np.bool_)
            mask[slice_x, slice_y] = True

            return DetectorInfo(
                shape=shape,
                mode=mode,
                slice=[slice_x, slice_y],
                mask=mask,
                geometry=RectangleGeometry(
                    xmin=slice_x.start,
                    xmax=slice_x.stop,
                    ymin=slice_y.start,
                    ymax=slice_y.stop,
                ),
            )

        case _:
            raise ValueError("Detector could not be determined")


def add_diffraction_roi(self: "DataViewer", item):
    """
    Attach a pre-built pyqtgraph item (an ROI such as ``pg.ROI``/``pg.LineROI``/
    ``pg.PolygonROI``, or a line/text/path) to the diffraction view.

    The plugin owns the item's lifetime; this only adds it to the view. The
    item is returned so a plugin can chain or re-store it. Detach with
    :func:`remove_diffraction_roi`.
    """
    self.diffraction_space_widget.getView().addItem(item)
    return item


def remove_diffraction_roi(self: "DataViewer", item):
    """
    Detach a previously-attached pyqtgraph item from the diffraction view (the
    inverse of :func:`add_diffraction_roi`). The item is *not* deleted, so a
    plugin may re-add it later.
    """
    self.diffraction_space_widget.getView().removeItem(item)
    return item


def add_real_space_roi(self: "DataViewer", item):
    """
    Attach a pre-built pyqtgraph item (an ROI such as ``pg.ROI``/``pg.LineROI``/
    ``pg.PolygonROI``, or a line/text/path) to the built-in real-space (virtual
    image) view — the default tab of the virtual-image pane.

    The plugin owns the item's lifetime; this only adds it to the view. The
    item is returned so a plugin can chain or re-store it. Detach with
    :func:`remove_real_space_roi`.
    """
    self.real_space_widget.getView().addItem(item)
    return item


def remove_real_space_roi(self: "DataViewer", item):
    """
    Detach a previously-attached pyqtgraph item from the built-in real-space
    (virtual image) view (the inverse of :func:`add_real_space_roi`). The item
    is *not* deleted, so a plugin may re-add it later.
    """
    self.real_space_widget.getView().removeItem(item)
    return item


def update_realspace_detector(self: "DataViewer"):
    # change the shape of the detector, then update the view

    detector_shape = (
        self.rs_detector_shape_group.checkedAction().text().replace("&", "")
    )
    assert detector_shape in ["Point", "Rectangular"], detector_shape

    main_pen = {"color": "g", "width": 6}
    handle_pen = {"color": "r", "width": 9}
    hover_pen = {"color": "c", "width": 6}
    hover_handle = {"color": "c", "width": 9}

    if self.unscaled_realspace_image is None:
        x0, y0 = 0, 0
        xr, yr = 4, 4
    else:
        x, y = self.unscaled_realspace_image.shape[:2]
        y0, x0 = x // 2, y // 2
        xr, yr = (np.minimum(x, y) / 10,) * 2

    # Remove existing detector
    if hasattr(self, "real_space_point_selector"):
        self.real_space_widget.view.scene().removeItem(self.real_space_point_selector)
        self.real_space_point_selector = None
    if hasattr(self, "real_space_rect_selector"):
        self.real_space_widget.view.scene().removeItem(self.real_space_rect_selector)
        self.real_space_rect_selector = None

    # Point detector
    if detector_shape == "Point":
        self.real_space_point_selector = pg_point_roi(
            self.real_space_widget.getView(),
            center=(x0 - 0.5, y0 - 0.5),
            pen=main_pen,
            hoverPen=hover_pen,
        )
        self.real_space_point_selector.sigRegionChanged.connect(
            partial(self.update_diffraction_space_view, False)
        )

    # Rectangular detector
    elif detector_shape == "Rectangular":
        self.real_space_rect_selector = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.real_space_widget.getView().addItem(self.real_space_rect_selector)
        self.real_space_rect_selector.sigRegionChangeFinished.connect(
            partial(self.update_diffraction_space_view, False)
        )

    else:
        raise ValueError("Unknown detector shape! Got: {}".format(detector_shape))

    self.update_diffraction_space_view(reset=True)


def update_diffraction_detector(self: "DataViewer"):
    # change the shape of the detector, then update the view

    detector_shape = self.detector_shape_group.checkedAction().text().strip("&")
    assert detector_shape in ["Point", "Rectangular", "Circle", "Annulus"]

    main_pen = {"color": "g", "width": 6}
    handle_pen = {"color": "r", "width": 9}
    hover_pen = {"color": "c", "width": 6}
    hover_handle = {"color": "c", "width": 9}

    if self.unscaled_diffraction_image is None:
        x0, y0 = 0, 0
        xr, yr = 4, 4
    else:
        x, y = self.unscaled_diffraction_image.shape[:2]
        y0, x0 = x // 2, y // 2
        xr, yr = (np.minimum(x, y) / 10,) * 2

    # Remove existing detector
    if hasattr(self, "virtual_detector_point"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_point
        )
        self.virtual_detector_point = None
    if hasattr(self, "virtual_detector_roi"):
        self.diffraction_space_widget.view.scene().removeItem(self.virtual_detector_roi)
        self.virtual_detector_roi = None
    if hasattr(self, "virtual_detector_roi_inner"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_inner
        )
        self.virtual_detector_roi_inner = None
    if hasattr(self, "virtual_detector_roi_outer"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_outer
        )
        self.virtual_detector_roi_outer = None

    # Point detector
    if detector_shape == "Point":
        self.virtual_detector_point = pg_point_roi(
            self.diffraction_space_widget.getView(),
            center=(x0 - 0.5, y0 - 0.5),
            pen=main_pen,
            hoverPen=hover_pen,
        )
        self.virtual_detector_point.sigRegionChanged.connect(
            partial(self.update_real_space_view, False)
        )

    # Rectangular detector
    elif detector_shape == "Rectangular":
        self.virtual_detector_roi = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    # Circular detector
    elif detector_shape == "Circle":
        self.virtual_detector_roi = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    # Annular dector
    elif detector_shape == "Annulus":
        # Make outer detector
        self.virtual_detector_roi_outer = pg.CircleROI(
            [int(x0 - xr), int(y0 - yr)],
            [int(2 * xr), int(2 * yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_outer)

        # Make inner detector
        self.virtual_detector_roi_inner = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            hoverPen=hover_pen,
            handlePen=handle_pen,
            handleHoverPen=hover_handle,
            movable=False,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_inner)

        # Connect size/position of inner and outer detectors
        self.virtual_detector_roi_outer.sigRegionChanged.connect(
            self.update_annulus_pos
        )
        self.virtual_detector_roi_outer.sigRegionChanged.connect(
            self.update_annulus_radii
        )
        self.virtual_detector_roi_inner.sigRegionChanged.connect(
            self.update_annulus_radii
        )

        # Connect to real space view update function
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )
        self.virtual_detector_roi_inner.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    else:
        raise ValueError("Unknown detector shape! Got: {}".format(detector_shape))

    self.update_real_space_view(reset=True)


def nudge_real_space_selector(self: "DataViewer", dx, dy):
    selector = _real_space_selector(self)

    position = selector.pos()
    position[0] += dy
    position[1] += dx

    selector.setPos(position)


def nudge_diffraction_selector(self: "DataViewer", dx, dy):
    selector = _diffraction_selector(self)

    position = selector.pos()
    position[0] += dy
    position[1] += dx

    selector.setPos(position)


def _diffraction_selector(self: "DataViewer"):
    """The currently-active diffraction-space ROI (point / rect / circle / annulus)."""
    if (
        hasattr(self, "virtual_detector_point")
        and self.virtual_detector_point is not None
    ):
        return self.virtual_detector_point
    if hasattr(self, "virtual_detector_roi") and self.virtual_detector_roi is not None:
        return self.virtual_detector_roi
    if (
        hasattr(self, "virtual_detector_roi_outer")
        and self.virtual_detector_roi_outer is not None
    ):
        return self.virtual_detector_roi_outer
    raise RuntimeError("Can't find the diffraction space selector!")


def _real_space_selector(self: "DataViewer"):
    """The currently-active real-space (virtual-image) ROI (point / rect)."""
    if (
        hasattr(self, "real_space_point_selector")
        and self.real_space_point_selector is not None
    ):
        return self.real_space_point_selector
    if (
        hasattr(self, "real_space_rect_selector")
        and self.real_space_rect_selector is not None
    ):
        return self.real_space_rect_selector
    raise RuntimeError("Can't find the real space selector!")


def center_diffraction_selector_at(self: "DataViewer", view_pos):
    """Recenter the active diffraction-space ROI on ``view_pos`` (in the view's
    local/data coordinate space). ``setPos`` fires ``sigRegionChanged`` /
    ``sigRegionChangeFinished``, so the real-space view recomputes just as it does
    for a drag or a keyboard nudge — no explicit view update is needed."""
    selector = _diffraction_selector(self)
    size = selector.size()
    selector.setPos((view_pos.x() - size.x() / 2.0, view_pos.y() - size.y() / 2.0))


def center_real_space_selector_at(self: "DataViewer", view_pos):
    """Recenter the active real-space ROI on ``view_pos`` (in the view's
    local/data coordinate space). See :func:`center_diffraction_selector_at`."""
    selector = _real_space_selector(self)
    size = selector.size()
    selector.setPos((view_pos.x() - size.x() / 2.0, view_pos.y() - size.y() / 2.0))


def on_diffraction_double_click(self: "DataViewer", event):
    """Diffraction-pane scene click handler: on a left-button double-click, center
    the detector on the clicked data point. Single clicks, drags, and
    right/middle-button double-clicks are ignored."""
    if not event.double() or event.button() != QtCore.Qt.LeftButton:
        return
    view = self.diffraction_space_widget.view
    view_pos = view.mapSceneToView(event.scenePos())
    self.center_diffraction_selector_at(view_pos)


def on_real_space_double_click(self: "DataViewer", event):
    """Virtual-image (real-space) pane scene click handler: on a left-button
    double-click, center the scan-position selector on the clicked data point."""
    if not event.double() or event.button() != QtCore.Qt.LeftButton:
        return
    view = self.real_space_widget.view
    view_pos = view.mapSceneToView(event.scenePos())
    self.center_real_space_selector_at(view_pos)


def update_annulus_pos(self: "DataViewer"):
    """
    Function to keep inner and outer rings of annulus aligned.
    """
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    # Only outer annulus is draggable; when it moves, update position of inner annulus
    x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
    y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
    self.virtual_detector_roi_inner.setPos(x0 - R_inner, y0 - R_inner, update=False)


def update_annulus_radii(self: "DataViewer"):
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    if R_outer < R_inner:
        x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
        y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
        self.virtual_detector_roi_outer.setSize(2 * R_inner + 6, update=False)
        self.virtual_detector_roi_outer.setPos(
            x0 - R_inner - 3, y0 - R_inner - 3, update=False
        )
