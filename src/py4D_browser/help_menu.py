"""
The in-app keyboard map.

Renders a keyboard layout with the keys py4DGUI binds, highlighted and
labeled, plus a list of the Ctrl-based menu shortcuts. Everything is plain
Qt widgets — crisp at any window size, no external image to keep in sync
with the code.

Color coding (matching the old pre-rendered keymap), explained by the
legend under the keyboard:
  coral  W A S D  — nudge the diffraction-space detector
  blue   I J K L  — nudge the real-space scan position
  green  SHIFT    — hold to move 5 px per press instead of 1 px

A double-click in the diffraction or virtual-image pane also jumps the
selector to that data point (see the mouse note in the legend).
"""

import platform

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

IS_MAC = platform.system() == "Darwin"

DETECTOR_COLOR = "#e4715e"  # W/A/S/D — detector nudge
SELECTOR_COLOR = "#4c9fd8"  # I/J/K/L — real-space selector nudge
SHIFT_COLOR = "#52b788"  # SHIFT — 5 px per press
DOUBLE_CLICK_COLOR = "#9b6dd6"  # double-click — jump selector to a point

_NORMAL_STYLE = """
    QLabel {
        background-color: #f6f6f6;
        color: #333333;
        border: 1px solid #a8a8a8;
        border-bottom: 3px solid #8a8a8a;
        border-radius: 4px;
        font-size: 11px;
    }
"""


def _highlight_style(color):
    return f"""
    QLabel {{
        background-color: {color};
        color: white;
        border: 1px solid rgba(0, 0, 0, 0.25);
        border-bottom: 3px solid rgba(0, 0, 0, 0.4);
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }}
"""


class KeyCap(QLabel):
    """
    A single keycap: a key label, optionally with a short description below
    it and an accent color. All caps of a given size are fixed-height, so
    rows line up regardless of what text a highlighted key carries.
    """

    def __init__(
        self,
        label: str,
        description: str = None,
        accent: str = None,
        small: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        if description is not None:
            self.setText(
                f'<div style="font-size:12px;font-weight:600;">{label}</div>'
                f'<div style="font-size:8px;">{description}</div>'
            )
        else:
            self.setText(label)
        if accent is not None:
            self.setStyleSheet(_highlight_style(accent))
        else:
            self.setStyleSheet(_NORMAL_STYLE)
        self.setFixedHeight(22 if small else 44)

    def minimumSizeHint(self):
        # Keep the minimum identical for every cap so a row distributes its
        # width purely by stretch factor. Otherwise a key's own text —
        # especially the two-line highlighted keys — sets a larger minimum
        # and the layout widens it past its neighbors, breaking the column
        # alignment between rows.
        return QtCore.QSize(20, 0)


# One entry per key: (label, relative width, description?, accent?).
# A ``None`` label is a fixed gap between key clusters.
_KEYBOARD_ROWS = [
    [
        ("ESC", 1),
        ("F1", 1),
        ("F2", 1),
        ("F3", 1),
        ("F4", 1),
        (None, 0.3),
        ("F5", 1),
        ("F6", 1),
        ("F7", 1),
        ("F8", 1),
        (None, 0.3),
        ("F9", 1),
        ("F10", 1),
        ("F11", 1),
        ("F12", 1),
    ],
    [
        ("`", 1),
        ("1", 1),
        ("2", 1),
        ("3", 1),
        ("4", 1),
        ("5", 1),
        ("6", 1),
        ("7", 1),
        ("8", 1),
        ("9", 1),
        ("0", 1),
        ("-", 1),
        ("=", 1),
        ("BACKSPACE", 2.1),
    ],
    [
        ("TAB", 1.5),
        ("Q", 1),
        ("W", 1, "▲", DETECTOR_COLOR),
        ("E", 1),
        ("R", 1),
        ("T", 1),
        ("Y", 1),
        ("U", 1),
        ("I", 1, "▲", SELECTOR_COLOR),
        ("O", 1),
        ("P", 1),
        ("[", 1),
        ("]", 1),
        ("\\", 1),
    ],
    [
        ("CAPS", 1.7),
        ("A", 1, "◀", DETECTOR_COLOR),
        ("S", 1, "▼", DETECTOR_COLOR),
        ("D", 1, "▶", DETECTOR_COLOR),
        ("F", 1),
        ("G", 1),
        ("H", 1),
        ("J", 1, "◀", SELECTOR_COLOR),
        ("K", 1, "▼", SELECTOR_COLOR),
        ("L", 1, "▶", SELECTOR_COLOR),
        (";", 1),
        ("'", 1),
        ("ENTER", 2.4),
    ],
    [
        ("SHIFT", 2.0, "5 px per press", SHIFT_COLOR),
        ("Z", 1),
        ("X", 1),
        ("C", 1),
        ("V", 1),
        ("B", 1),
        ("N", 1),
        ("M", 1),
        (",", 1),
        (".", 1),
        ("/", 1),
        ("SHIFT", 2.3, "5 px per press", SHIFT_COLOR),
    ],
    # The bottom rows differ by platform: a Mac has
    # Control-Option-Command, a Windows/Linux keyboard has Ctrl-Win-Alt.
    (
        [
            ("CTRL", 1.4),
            ("OPTION", 1.8),
            ("COMMAND", 2.2),
            ("SPACE", 4.8),
            ("COMMAND", 2.2),
            ("OPTION", 1.8),
            ("CTRL", 1.4),
        ]
        if IS_MAC
        else [
            ("CTRL", 1.4),
            ("WIN", 1.4),
            ("ALT", 1.4),
            ("SPACE", 7.4),
            ("ALT", 1.4),
            ("MENU", 1.4),
            ("CTRL", 1.4),
        ]
    ),
]

# "Alt" is the Option key, and the Ctrl modifier is the Command key, on a
# Mac; spell both as Mac users know them.
_ALT = "Option" if IS_MAC else "Alt"
_CTRL = "Command" if IS_MAC else "Ctrl"

# Tab switching uses the physical Control key ("Ctrl") on Windows/Linux, but
# the Option key on a Mac (where Control/Command + Tab arrive as a codepoint
# that a Key_Tab shortcut won't match, while Option + Tab arrives cleanly).
_TAB_MOD = "Option" if IS_MAC else "Ctrl"

# (keys, description) for each modifier-based menu shortcut.
_MENU_SHORTCUTS = [
    ([_CTRL, "O"], "Load data"),
    ([_CTRL, "S"], "Export datacube (py4DSTEM HDF5)"),
    ([_CTRL, "C"], "Copy virtual image to clipboard"),
    ([_CTRL, _ALT, "C"], "Copy diffraction pattern to clipboard"),
    ([_CTRL, "Shift", "C"], "Copy result to clipboard"),
    ([_TAB_MOD, "Tab"], "Next virtual-image tab"),
    ([_TAB_MOD, "Shift", "Tab"], "Previous virtual-image tab"),
    ([_CTRL, "Shift", "D"], "Debug console"),
    ([_CTRL, "W"], "Close window"),
]

_LEGEND = [
    (DETECTOR_COLOR, "W A S D — nudge the detector"),
    (SELECTOR_COLOR, "I J K L — nudge the scan position"),
    (SHIFT_COLOR, "Shift — 5 px per press"),
    (
        DOUBLE_CLICK_COLOR,
        "Double-click — jump the detector / selector to that point",
    ),
]


def _section_title(text: str):
    label = QLabel(f"<b style='font-size:13px;'>{text}</b>")
    return label


def _build_legend():
    row = QHBoxLayout()
    row.setSpacing(14)
    for color, text in _LEGEND:
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background-color:{color}; border-radius:3px;")
        row.addWidget(swatch)
        row.addWidget(QLabel(text))
    row.addStretch(1)
    return row


def _build_key_row(entries):
    row = QHBoxLayout()
    row.setSpacing(4)
    for entry in entries:
        label = entry[0]
        width = entry[1]
        description = entry[2] if len(entry) > 2 else None
        accent = entry[3] if len(entry) > 3 else None
        if label is None:
            row.addSpacing(8)
            continue
        key = KeyCap(label, description=description, accent=accent)
        row.addWidget(key, int(round(width * 10)))
    return row


def _build_shortcut_row(keys, description):
    row = QHBoxLayout()
    row.setSpacing(4)
    for i, key in enumerate(keys):
        if i:
            plus = QLabel("+")
            plus.setAlignment(QtCore.Qt.AlignCenter)
            row.addWidget(plus)
        row.addWidget(KeyCap(key, small=True))
    row.addSpacing(12)
    label = QLabel(description)
    row.addWidget(label)
    row.addStretch(1)
    return row


class KeyboardMapMenu(QDialog):
    """The Help → Keyboard Map dialog: a natively rendered keyboard map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(_section_title("Keys"))
        layout.addLayout(_build_legend())
        for entries in _KEYBOARD_ROWS:
            layout.addLayout(_build_key_row(entries))

        layout.addSpacing(12)
        layout.addWidget(_section_title("Menu shortcuts"))
        for keys, description in _MENU_SHORTCUTS:
            layout.addLayout(_build_shortcut_row(keys, description))

        layout.addStretch(1)
        self.resize(900, 560)
