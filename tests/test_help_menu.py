"""
Tests for the in-app keyboard map (Help → Keyboard Map).
"""


def test_keyboard_map_dialog_builds(qapp):
    from py4D_browser.help_menu import (
        KeyboardMapMenu,
        KeyCap,
        DETECTOR_COLOR,
        SELECTOR_COLOR,
        SHIFT_COLOR,
    )

    dlg = KeyboardMapMenu()
    assert dlg.windowTitle() == "Keyboard Shortcuts"

    caps = [w for w in dlg.findChildren(KeyCap) if not w.minimumHeight() == 22]
    highlighted = [c for c in caps if "rgba" in c.styleSheet()]
    assert len(highlighted) >= 9

    by_label = {}
    for cap in highlighted:
        # Rich text: the key label is the text of the first <div>.
        label = cap.text().split(">")[1].split("<")[0].strip()
        by_label.setdefault(label, set()).add(cap.styleSheet())

    # W/A/S/D are the detector nudge keys, I/J/K/L the selector nudge keys,
    # SHIFT the 5-px multiplier.
    for label in "WASD":
        assert DETECTOR_COLOR in next(iter(by_label[label])), label
    for label in "IJKL":
        assert SELECTOR_COLOR in next(iter(by_label[label])), label
    assert SHIFT_COLOR in next(iter(by_label["SHIFT"]))


def test_keyboard_map_lists_menu_shortcuts(qapp):
    from py4D_browser.help_menu import KeyboardMapMenu, KeyCap

    dlg = KeyboardMapMenu()
    small_caps = [w for w in dlg.findChildren(KeyCap) if w.minimumHeight() == 22]
    labels = [c.text() for c in small_caps]
    # Every shortcut row uses at least a Ctrl (Command on a Mac) key; the
    # full set spans the Tab pair, Alt/Option+C, and Shift+C/D.
    assert ("Ctrl" in labels or "Command" in labels) and "Tab" in labels
    assert "Shift" in labels and "D" in labels
    # "Alt" is the Option key on a Mac.
    assert "Option" in labels or "Alt" in labels
