#!/usr/bin/env python

import py4D_browser
import sys
from PyQt5.QtWidgets import QApplication


def launch():
    app = QApplication(sys.argv)
    win = py4D_browser.DataViewer(sys.argv)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
