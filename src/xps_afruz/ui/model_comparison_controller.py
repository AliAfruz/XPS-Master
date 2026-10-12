"""GUI lifecycle for retained physics-constrained model comparisons."""

from __future__ import annotations

from PySide6.QtCore import QThread

from ..backgrounds import diagnose_background
from ..fitting.model_comparison import (
    ModelComparisonResult,
    ModelComparisonSettings,
    build_standard_candidates,
    optimized_components,
)
from .model_comparison_report import open_model_comparison_report
from .model_comparison_worker import ModelComparisonWorker


class ModelComparisonControllerMixin:
    """Build, run, retain, inspect, and explicitly accept model alternatives."""

    def start_model_comparison(self) -> None:
        """Fit and rank controlled alternatives without replacing the active model."""

        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            self.show_error("Global fitting is running", "Finish or stop global multi-spectrum fitting first.")
            return
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            self.statusBar().showMessage("Model comparison is already running.")
            return
        if self._smart_thread is not None and self._smart_thread.isRunning():
            self.show_error("Smart search is running", "Finish or stop smart deconvolution before comparing models.")
            return
        if self._consensus_thread is not None and self._consensus_thread.isRunning():
            self.show_error("Consensus is running", "Finish or stop stability consensus before comparing models.")
            return
        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a high-resolution spectrum before comparing fit models.")
            return
        try:
            components = tuple(self.components_from_table())
            ties = tuple(self.active_ties())
            settings = ModelComparisonSettings(
                compare_backgrounds=self.comparison_backgrounds.isChecked(),
                compare_profiles=self.comparison_profiles.isChecked(),
                compare_neighbor_counts=self.comparison_neighbors.isChecked(),
                weighting=self.fit_weighting,
                count_scale=self.poisson_count_scale,
                max_nfev=int(self.comparison_max_nfev.value()),
            )
            candidates = build_standard_candidates(
                self.spectrum,
                components,
                ties,
                self.background_settings,
                settings,
            )
        except (TypeError, ValueError) as error:
            self.show_error("Model comparison could not start", str(error))
            return
        if len(candidates) < 2:
            self.show_error(
                "No distinct alternatives",
                "Enable at least one background, line-shape, or neighbouring-count comparison.",
            )
            return

        self._comparison_workspace_index = self._active_workspace_index
        self._comparison_target_spectrum = self.spectrum
        thread = QThread(self)
        worker = ModelComparisonWorker(self.spectrum, candidates, settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_model_comparison_progress)
        worker.completed.connect(self._apply_model_comparison_result)
        worker.failed.connect(self._model_comparison_failed)
        worker.cancelled.connect(self._model_comparison_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._model_comparison_finished)
        self._comparison_thread = thread
        self._comparison_worker = worker
        self._set_model_comparison_running(True)
        self.comparison_progress.setValue(0)
        self.comparison_progress.setFormat("0%")
        self.comparison_status.setText(
            f"Evaluating {len(candidates)} explicit candidates. The active fit remains unchanged until acceptance."
        )
        self.statusBar().showMessage(f"Model comparison started with {len(candidates)} candidates.", 8_000)
        thread.start()

    def cancel_model_comparison(self) -> None:
        if self._comparison_worker is None:
            return
        self._comparison_worker.cancel()
        self.comparison_cancel_button.setEnabled(False)
        self.comparison_status.setText("Stopping after the current bounded candidate fit finishes…")

    def _set_model_comparison_running(self, running: bool) -> None:
        self.comparison_start_button.setEnabled(not running)
        self.comparison_cancel_button.setEnabled(running)
        for widget in (
            self.comparison_backgrounds,
            self.comparison_profiles,
            self.comparison_neighbors,
            self.comparison_max_nfev,
        ):
            widget.setEnabled(not running)
        self.smart_start_button.setEnabled(not running)
        self.consensus_start_button.setEnabled(not running)

    def _update_model_comparison_progress(self, percentage: int, message: str) -> None:
        self.comparison_progress.setValue(percentage)
        self.comparison_progress.setFormat(f"{percentage}%")
        self.comparison_status.setText(message)

    def _apply_model_comparison_result(self, result: ModelComparisonResult) -> None:
        if (
            self.spectrum is not self._comparison_target_spectrum
            or self._active_workspace_index != self._comparison_workspace_index
        ):
            self.comparison_status.setText(
                "Comparison completed, but the active spectrum changed; results were not attached to the wrong card."
            )
            return
        self._model_comparison_result = result
        index = self._active_workspace_index
        if index is not None and 0 <= index < len(self._workspace_model_comparisons):
            self._workspace_model_comparisons[index] = result
        self.comparison_progress.setValue(100)
        self.comparison_progress.setFormat("100%")
        self.comparison_report_button.setEnabled(True)
        recommended = result.recommended
        self.comparison_status.setText(
            f"Comparison ready: {len(result.candidates)} retained candidates. Numerical recommendation: "
            f"{recommended.candidate.label} ({recommended.model_support:.1%} support). Review before acceptance."
        )
        self._refresh_workspace_views()
        self._set_project_dirty()
        self.statusBar().showMessage(
            "Model comparison complete. Open the dashboard; the current fit has not been replaced.",
            10_000,
        )

    def _model_comparison_failed(self, message: str) -> None:
        self.comparison_progress.setValue(0)
        self.comparison_progress.setFormat("Not completed")
        self.comparison_status.setText(f"Model comparison could not complete: {message}")
        self.statusBar().showMessage("Model comparison did not complete.", 8_000)

    def _model_comparison_cancelled(self) -> None:
        self.comparison_progress.setFormat("Stopped")
        if self._model_comparison_result is None:
            self.comparison_status.setText("Comparison stopped; no partial candidate set was retained.")
        else:
            self.comparison_status.setText("Comparison stopped; the previous completed candidate set remains available.")

    def _model_comparison_finished(self) -> None:
        self._comparison_thread = None
        self._comparison_worker = None
        self._set_model_comparison_running(False)

    def show_model_comparison_report(self) -> None:
        if self.spectrum is None or self._model_comparison_result is None:
            self.show_error("No model comparison", "Run model comparison for the active spectrum first.")
            return
        open_model_comparison_report(
            self,
            self.spectrum,
            self._model_comparison_result,
            on_accept=self.accept_model_comparison_candidate,
            install_table_menu=self._install_table_copy_menu,
        )

    def accept_model_comparison_candidate(self, identifier: str) -> None:
        """Replace the active model only after an explicit dashboard action."""

        result = self._model_comparison_result
        if result is None or self.spectrum is None:
            self.show_error("No retained comparison", "Run model comparison before accepting a candidate.")
            return
        evaluation = result.candidate(identifier)
        if not evaluation.eligible:
            self.show_error("Candidate is not eligible", "This candidate failed the fit, identifiability, or background checks. Review its dashboard warnings before building a revised model.")
            return
        self._applying_model_comparison = True
        try:
            method = evaluation.candidate.background.method
            self.background_combo.blockSignals(True)
            self.background_combo.setCurrentIndex(self.background_combo.findData(method.value))
            self.background_combo.blockSignals(False)
            self._background_parameters = dict(evaluation.candidate.background.parameters)
            self.background = evaluation.fit_result.background.copy()
            self.background_diagnostics = diagnose_background(self.spectrum, self.background)
            self.background_info.setText(f"{self._background_description()}\n{self._background_diagnostic_summary()}")
            self.peak_table.setRowCount(0)
            for component in optimized_components(evaluation):
                self.add_peak_row(component)
            self._ties = list(evaluation.candidate.ties)
            self.fit_result = evaluation.fit_result
            self._update_table_values(self.fit_result.parameter_values)
            index = self._active_workspace_index
            if index is not None:
                self._workspace_background_settings[index] = (method, dict(self._background_parameters))
            self._invalidate_active_consensus(
                "Accepted model changed the fit hypothesis; rerun stability consensus for updated evidence."
            )
            self.fit_summary.setText(
                f"<b>Accepted comparison candidate</b><br>"
                f"{evaluation.candidate.label} · rank {evaluation.rank} · support {evaluation.model_support:.1%}<br>"
                f"{self._fit_quality_lines(evaluation.fit_result)}<br>"
                f"Background: {self._background_description()}<br>"
                f"{evaluation.identifiability}"
            )
            self.render_spectrum()
            self._store_active_workspace_fit_state()
        finally:
            self._applying_model_comparison = False
        self.statusBar().showMessage(
            f"Accepted {evaluation.candidate.label}; all comparison alternatives remain retained.",
            10_000,
        )
