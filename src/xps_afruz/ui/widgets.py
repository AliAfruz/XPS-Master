"""Small reusable Qt controls with safe scientific-input behavior."""

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    """Keep panel scrolling from changing a focused selection."""

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Require deliberate typing or arrow clicks for decimal values."""

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Require deliberate typing or arrow clicks for integer values."""

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        event.ignore()
