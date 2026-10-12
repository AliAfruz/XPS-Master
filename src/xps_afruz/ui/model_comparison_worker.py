"""Background worker for responsive physics-constrained model comparison."""

from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from ..core.spectrum import Spectrum
from ..fitting.model_comparison import (
    ModelCandidate,
    ModelComparisonCancelled,
    ModelComparisonResult,
    ModelComparisonSettings,
    compare_models,
)


class ModelComparisonWorker(QObject):
    """Run one immutable candidate batch outside the Qt event loop."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        spectrum: Spectrum,
        candidates: tuple[ModelCandidate, ...],
        settings: ModelComparisonSettings,
    ) -> None:
        super().__init__()
        self._spectrum = spectrum
        self._candidates = candidates
        self._settings = settings
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            result = compare_models(
                self._spectrum,
                self._candidates,
                self._settings,
                progress_callback=self._report_progress,
                cancellation_callback=self._cancelled.is_set,
            )
        except ModelComparisonCancelled:
            self.cancelled.emit()
        except Exception as error:  # Qt worker boundary reports a concise user-visible failure.
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)
        finally:
            self.finished.emit()

    def _report_progress(self, completed: int, total: int, message: str) -> None:
        percentage = int(round(100.0 * completed / max(total, 1)))
        self.progress.emit(percentage, message)
