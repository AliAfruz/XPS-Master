"""Metallic-dark Qt stylesheet shared by the XPS-master desktop interface."""

SPACE_STYLESHEET = """
QMainWindow {
    background: #151718;
    color: #e6e8e6;
    font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif;
}
QWidget { color: #dde1df; font-size: 12px; }
QWidget#controlPanel {
    background: #1c1f20;
    border: 1px solid #34393b;
    border-radius: 10px;
}
QFrame#glassPanel, QFrame#plotCard {
    background: #232728;
    border: 1px solid #3a4042;
    border-radius: 8px;
}
QLabel#eyebrow { color: #d2a55f; font-size: 10px; font-weight: 700; letter-spacing: 1.4px; }
QLabel#appTitle { color: #f3f2ed; font-size: 24px; font-weight: 700; }
QLabel#subtitle { color: #99a3a2; font-size: 11px; }
QLabel#sectionTitle { color: #ece8df; font-size: 12px; font-weight: 700; }
QLabel#plotKicker { color: #cfa45f; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; }
QLabel#plotTitle { color: #f0f1ee; font-size: 14px; font-weight: 600; }
QLabel#enginePill {
    color: #a9d8c4; background: #21352f; border: 1px solid #356153;
    border-radius: 10px; padding: 4px 9px; font-size: 10px; font-weight: 700;
}
QLabel#fitSummary {
    color: #dce1de; background: #191c1d; border: 1px solid #373d3f;
    border-radius: 7px; padding: 11px;
}
QPushButton {
    background: #303537;
    color: #f0f1ed;
    border: 1px solid #484f52;
    border-radius: 6px;
    min-height: 34px;
    padding: 0px 12px;
    font-weight: 600;
}
QPushButton:hover { background: #3a4042; border-color: #c99d59; color: #ffffff; }
QPushButton:pressed { background: #25292a; border-color: #9b7744; }
QPushButton:focus { border: 1px solid #d6aa64; }
QPushButton:disabled { background: #242829; color: #707879; border-color: #34393b; }
QPushButton#primaryButton {
    background: #c3924c;
    border: 1px solid #e0b778;
    color: #17191a;
    min-height: 40px;
    font-size: 13px;
    font-weight: 700;
}
QPushButton#primaryButton:hover { background: #d1a35d; border-color: #efd09e; }
QPushButton#removeButton { color: #f3d6d8; background: #533034; border-color: #77454b; }
QPushButton#removeButton:hover { background: #65383e; border-color: #ba6e77; }
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background: #171a1b;
    border: 1px solid #444b4d;
    border-radius: 6px;
    min-height: 32px;
    padding: 0px 9px;
    color: #eef0ed;
    selection-background-color: #725d3d;
}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #8b7655; }
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus { border-color: #d2a55f; }
QComboBox::drop-down { border: 0px; width: 26px; }
QComboBox QAbstractItemView { background: #202425; color: #ecf0ed; border: 1px solid #4b5355; selection-background-color: #725d3d; padding: 4px; }
QTableWidget {
    background: #181b1c;
    alternate-background-color: #202425;
    border: 1px solid #3b4143;
    border-radius: 6px;
    gridline-color: #303638;
    selection-background-color: #5d5140;
    selection-color: #ffffff;
    padding: 2px;
}
QTableWidget::item { padding: 6px 5px; border: 0px; }
QHeaderView::section {
    background: #2b3032;
    color: #e8dfcf;
    border: 0px;
    border-right: 1px solid #3b4244;
    border-bottom: 1px solid #6d5839;
    padding: 8px 5px;
    font-size: 10px;
    font-weight: 700;
}
QTableCornerButton::section { background: #2b3032; border: 0px; }
QListWidget { background: #181b1c; alternate-background-color: #202425; border: 1px solid #3b4143; border-radius: 6px; padding: 4px; outline: 0px; }
QListWidget::item { border-radius: 4px; padding: 8px; margin: 1px; }
QListWidget::item:selected { background: #5d5140; color: #ffffff; }
QListWidget::item:hover { background: #2b3032; }
QTabWidget::pane { border: 1px solid #353b3d; border-radius: 7px; top: -1px; background: #191c1d; }
QTabBar::tab { background: transparent; color: #98a2a1; border: 0px; border-bottom: 2px solid transparent; padding: 9px 14px; margin-right: 4px; font-weight: 600; }
QTabWidget#workflowTabs QTabBar::tab { padding: 9px 7px; margin-right: 1px; }
QWidget#workflowPage { background: #191c1d; }
QTabBar::tab:selected { color: #f1e7d6; border-bottom-color: #d0a158; }
QTabBar::tab:hover { color: #ffffff; background: #25292a; }
QMenuBar { background: #111314; color: #e3e6e4; border-bottom: 1px solid #303537; }
QMenuBar::item { padding: 7px 11px; background: transparent; }
QMenuBar::item:selected { background: #292e30; }
QMenu { background: #1d2021; color: #e9ece9; border: 1px solid #454c4e; padding: 5px; }
QMenu::item { padding: 7px 24px 7px 10px; border-radius: 4px; }
QMenu::item:selected { background: #68573e; }
QStatusBar { background: #111314; color: #a8b0ae; border-top: 1px solid #303537; min-height: 24px; }
QSplitter::handle { background: #2b3032; width: 4px; height: 4px; }
QSplitter::handle:hover { background: #ad844c; }
QDockWidget { background: #1b1e1f; color: #e7ebe9; border: 1px solid #343a3c; }
QDockWidget::title { background: #292e30; color: #ead7b7; padding: 10px 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; }
QWidget#surveyPanel { background: #1b1e1f; }
QLabel#surveySummary { color: #dce1de; background: #262b2d; border: 1px solid #3d4446; border-radius: 6px; padding: 10px; }
QLabel#surveyNote { color: #aeb7b5; font-size: 11px; line-height: 1.25; }
QCheckBox { color: #dfe5e2; spacing: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #677174; border-radius: 4px; background: #151819; }
QCheckBox::indicator:hover { border-color: #c99d59; }
QCheckBox::indicator:checked { background: #c4934d; border-color: #e0b778; }
QProgressBar { background: #151819; border: 1px solid #3d4446; border-radius: 5px; min-height: 18px; text-align: center; color: #f1ede2; }
QProgressBar::chunk { background: #b78a4a; border-radius: 4px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 3px; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 3px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #50585a; border-radius: 4px; min-height: 30px; min-width: 30px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #7b6748; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QToolTip { color: #f2f2ee; background: #272b2c; border: 1px solid #5a6264; padding: 6px; }
"""
