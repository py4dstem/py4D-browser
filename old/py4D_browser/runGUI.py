#!/usr/bin/env python

import py4D_browser
#from gui.viewer import DataViewer
import sys

def launch():
    app = py4D_browser.DataViewer(sys.argv)

    sys.exit(app.exec_())

if __name__ == '__main__':
    app = py4D_browser.DataViewer(sys.argv)

    sys.exit(app.exec_())
