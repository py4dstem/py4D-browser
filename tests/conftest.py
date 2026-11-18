"""
Shared fixtures for the py4D_browser test suite.

The tests run headless; Qt uses the offscreen platform plugin.
"""

import os

# Ensure Qt runs offscreen even if the user has not set this already.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def viewer(qapp):
    """
    A fully constructed DataViewer, with the real sample plugins loaded,
    pointing at a throwaway config file so the tests never touch the
    user's real GUI settings.
    """
    import py4D_browser.main_window as mw
    from py4D_browser import DataViewer

    config_dir = tempfile.mkdtemp(prefix="py4dgui_test_")
    # Route the GUI config file into a throwaway dir; undo on teardown so the
    # global `platformdirs` module attribute does not leak into other tests.
    mp = pytest.MonkeyPatch()
    mp.setattr(
        mw.platformdirs,
        "user_config_dir",
        lambda appname, appauthor=None, roaming=False: config_dir,
    )
    try:
        v = DataViewer(reset_state=True)
        yield v
        # Close any tabs a test left open (most close their own).
        for tab in list(getattr(v, "virtual_image_tabs", [])):
            v.close_virtual_image_tab(tab)
        # Destroy the C++ widget tree NOW, while the interpreter is still fully up.
        #
        # v.close() would only hide the window, leaving every C++ ViewBox (the
        # built-in pane's plus any tab's) alive for CPython to free in arbitrary
        # order at shutdown. pyqtgraph 0.14.0 then tears them down in a bad order:
        # one ViewBox's destroyed handler calls updateAllViewLists(), which touches
        # a half-freed sibling ViewBox and the process SIGSEGVs (exit 139) *after*
        # the tests have all passed, masking the pytest summary line. This is a
        # pre-existing pyqtgraph teardown-ordering bug (it reproduces on the
        # baseline, not on this branch's changes).
        #
        # deleteLater() + processEvents() makes Qt destroy the tree in its own
        # safe children-first order while the interpreter is alive, so there is
        # nothing left for the shutdown GC to tear down.
        v.deleteLater()
        qapp.processEvents()
    finally:
        mp.undo()
