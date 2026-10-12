"""GUI lifecycle for retained global multi-spectrum fitting."""

from __future__ import annotations

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from ..backgrounds import estimate_background
from ..fitting.global_series import GlobalSeriesResult, GlobalSeriesSettings
from .global_series_report import open_global_series_report
from .global_series_worker import GlobalSeriesWorker


class GlobalSeriesControllerMixin:
    """Select, fit, retain, inspect, and explicitly apply one spectral series."""

    def start_global_series_fit(self) -> None:
        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            self.statusBar().showMessage("A global series fit is already running.")
            return
        for label, thread in (
            ("smart deconvolution", self._smart_thread),
            ("stability consensus", self._consensus_thread),
            ("model comparison", self._comparison_thread),
        ):
            if thread is not None and thread.isRunning():
                self.show_error("Another fit is running", f"Finish or stop {label} before global fitting.")
                return
        if self.spectrum is None or self._active_workspace_index is None:
            self.show_error("No active fit hypothesis", "Load and activate a high-resolution spectrum first.")
            return
        if len(self._workspace_spectra) < 2:
            self.show_error("A series needs multiple spectra", "Load at least two related high-resolution spectra.")
            return
        try:
            self._synchronize_active_workspace()
            components = tuple(self.components_from_table())
            ties = tuple(self.active_ties())
            selected_indices = self._choose_global_series_indices(components)
            if selected_indices is None:
                return
            settings = GlobalSeriesSettings(
                share_centers=True,
                share_fwhm=self.global_share_fwhm.isChecked(),
                share_eta=self.global_share_shape.isChecked(),
                share_asymmetry=self.global_share_shape.isChecked(),
                share_instrument_fwhm=self.global_share_shape.isChecked(),
                allow_energy_shifts=self.global_allow_shifts.isChecked(),
                maximum_energy_shift_eV=float(self.global_shift_limit.value()),
                weighting=self.fit_weighting,
                count_scale=self.poisson_count_scale,
                max_nfev=int(self.global_max_nfev.value()),
            )
            spectra = tuple(self._workspace_spectra[index] for index in selected_indices)
            backgrounds = tuple(
                estimate_background(
                    self._workspace_spectra[index],
                    self._workspace_background_settings[index][0],
                    **dict(self._workspace_background_settings[index][1]),
                )
                for index in selected_indices
            )
        except (TypeError, ValueError) as error:
            self.show_error("Global fit could not start", str(error))
            return

        self._global_series_input_spectra = spectra
        thread = QThread(self)
        worker = GlobalSeriesWorker(
            spectra,
            backgrounds,
            components,
            ties,
            selected_indices,
            settings,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_global_series_progress)
        worker.completed.connect(self._apply_global_series_result)
        worker.failed.connect(self._global_series_failed)
        worker.cancelled.connect(self._global_series_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._global_series_finished)
        self._global_series_thread = thread
        self._global_series_worker = worker
        self._set_global_series_running(True)
        self.global_progress.setValue(0)
        self.global_progress.setFormat("0%")
        self.global_status.setText(
            f"Fitting {len(spectra)} spectra independently, then as one shared-parameter hypothesis."
        )
        self.statusBar().showMessage("Global multi-spectrum fitting started in a background worker.", 8_000)
        thread.start()

    def _choose_global_series_indices(self, components) -> tuple[int, ...] | None:  # type: ignore[no-untyped-def]
        """Ask which compatible workspace cards belong to the physical series."""

        active = self._active_workspace_index
        if active is None or self.spectrum is None:
            return None
        dialog = QDialog(self)
        dialog.setWindowTitle("Select spectra for global fitting")
        dialog.resize(580, 480)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "The active spectrum defines the peak hypothesis and reference energy. Select only spectra from the same "
            "high-resolution region and acquisition series. Each spectrum keeps its own background and peak areas."
        )
        note.setObjectName("surveyNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        listing = QListWidget()
        listing.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        compatible_count = 0
        centers = tuple(component.center for component in components)
        for index, spectrum in enumerate(self._workspace_spectra):
            compatible = (
                spectrum.energy_scale is self.spectrum.energy_scale
                and all(spectrum.energy_range[0] <= center <= spectrum.energy_range[1] for center in centers)
            )
            text = (
                f"{index + 1}. {spectrum.label} — {spectrum.energy_range[0]:.3f}–"
                f"{spectrum.energy_range[1]:.3f} eV, {spectrum.size} points"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if compatible else Qt.CheckState.Unchecked)
            if not compatible:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setToolTip("This spectrum does not cover every active component center or uses another energy scale.")
            else:
                compatible_count += 1
            if index == active:
                item.setCheckState(Qt.CheckState.Checked)
                item.setToolTip("Required reference spectrum: active peak hypothesis and zero relative shift.")
            listing.addItem(item)
        layout.addWidget(listing)
        status = QLabel(f"{compatible_count} spectrum/spectra are compatible with the active peak centers.")
        status.setObjectName("surveyNote")
        layout.addWidget(status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        selected = tuple(
            int(item.data(Qt.ItemDataRole.UserRole))
            for row in range(listing.count())
            if (item := listing.item(row)).checkState() == Qt.CheckState.Checked
        )
        if active not in selected:
            raise ValueError("The active reference spectrum must remain selected.")
        if len(selected) < 2:
            raise ValueError("Select at least two compatible spectra for global fitting.")
        # Put the active card first so its fitted energy shift is exactly zero.
        return (active,) + tuple(index for index in selected if index != active)

    def cancel_global_series_fit(self) -> None:
        if self._global_series_worker is None:
            return
        self._global_series_worker.cancel()
        self.global_cancel_button.setEnabled(False)
        self.global_status.setText("Stopping the optimizer safely; the previous retained result remains unchanged…")

    def _set_global_series_running(self, running: bool) -> None:
        self.global_start_button.setEnabled(not running and len(self._workspace_spectra) >= 2)
        self.global_cancel_button.setEnabled(running)
        for widget in (
            self.global_share_fwhm,
            self.global_share_shape,
            self.global_allow_shifts,
            self.global_shift_limit,
            self.global_max_nfev,
        ):
            widget.setEnabled(not running)
        self.smart_start_button.setEnabled(not running)
        self.consensus_start_button.setEnabled(not running)
        self.comparison_start_button.setEnabled(not running)

    def _update_global_series_progress(self, percentage: int, message: str) -> None:
        self.global_progress.setValue(percentage)
        self.global_progress.setFormat(f"{percentage}%")
        self.global_status.setText(message)

    def _apply_global_series_result(self, result: GlobalSeriesResult) -> None:
        current = tuple(
            self._workspace_spectra[index]
            for index in result.workspace_indices
            if 0 <= index < len(self._workspace_spectra)
        )
        if len(current) != len(self._global_series_input_spectra) or any(
            current_spectrum is not original_spectrum
            for current_spectrum, original_spectrum in zip(current, self._global_series_input_spectra, strict=True)
        ):
            self.global_status.setText(
                "Global calculation completed, but one selected spectrum changed; the result was not attached to stale data."
            )
            return
        self._global_series_result = result
        self.global_progress.setValue(100)
        self.global_progress.setFormat("100%")
        self.global_report_button.setEnabled(True)
        self.global_status.setText(
            f"Retained {len(result.spectrum_fits)}-spectrum result. {result.preference}. "
            f"ΔAICc {result.delta_aicc:+.2f}; ΔBIC {result.delta_bic:+.2f}."
        )
        self._set_project_dirty()
        self.statusBar().showMessage(
            "Global fit complete. Open the dashboard to inspect fits, residuals, shifts, and component trends.",
            10_000,
        )

    def _global_series_failed(self, message: str) -> None:
        self.global_progress.setValue(0)
        self.global_progress.setFormat("Not completed")
        self.global_status.setText(f"Global fit could not complete: {message}")
        self.statusBar().showMessage("Global multi-spectrum fit did not complete.", 8_000)

    def _global_series_cancelled(self) -> None:
        self.global_progress.setFormat("Stopped")
        if self._global_series_result is None:
            self.global_status.setText("Global fit stopped; no partial result was retained.")
        else:
            self.global_status.setText("Global fit stopped; the previous completed series result remains available.")

    def _global_series_finished(self) -> None:
        self._global_series_thread = None
        self._global_series_worker = None
        self._global_series_input_spectra = ()
        self._set_global_series_running(False)

    def show_global_series_report(self) -> None:
        result = self._global_series_result
        if result is None:
            self.show_error("No global-series result", "Run global multi-spectrum fitting first.")
            return
        spectra = {
            index: self._workspace_spectra[index]
            for index in result.workspace_indices
            if 0 <= index < len(self._workspace_spectra)
        }
        if len(spectra) != len(result.spectrum_fits):
            self.show_error("Series changed", "One or more spectra used by this retained result are no longer present.")
            return
        open_global_series_report(
            self,
            spectra,
            result,
            on_apply=self.apply_global_series_to_workspace,
            install_table_menu=self._install_table_copy_menu,
        )

    def apply_global_series_to_workspace(self) -> None:
        """Explicitly place globally fitted curves and component values on their cards."""

        result = self._global_series_result
        if result is None:
            return
        if not result.success:
            self.show_error("Global fit did not converge", "Review the retained diagnostics and revise the model before applying it to workspace cards.")
            return
        self._applying_global_series = True
        try:
            for series_position, item in enumerate(result.spectrum_fits):
                index = item.workspace_index
                if not 0 <= index < len(self._workspace_spectra):
                    continue
                rows = []
                for trend in result.component_trends:
                    rows.append(
                        {
                            "name": trend.name,
                            "center": trend.centers_eV[series_position],
                            "seed_center": trend.centers_eV[series_position],
                            "fwhm": trend.fwhm_eV[series_position],
                            "area": trend.areas[series_position],
                            "profile": trend.profile,
                            "eta": trend.eta[series_position],
                            "asymmetry": trend.asymmetry[series_position],
                            "instrument_fwhm": trend.instrument_fwhm_eV[series_position],
                            "vary_eta": trend.vary_eta,
                            "vary_asymmetry": trend.vary_asymmetry,
                            "vary_instrument_fwhm": trend.vary_instrument_fwhm,
                        }
                    )
                self._workspace_peak_rows[index] = rows
                self._workspace_ties[index] = list(result.ties)
                self._workspace_fit_results[index] = item.global_fit
                self._workspace_consensus_results[index] = None
                self._workspace_model_comparisons[index] = None
            active = self._active_workspace_index
            if active is not None and active in result.workspace_indices:
                self._restore_workspace_fit_state(active)
                self._restore_workspace_consensus_state(active)
                self._restore_workspace_model_comparison_state(active)
            self._refresh_workspace_views()
            self._set_project_dirty()
        finally:
            self._applying_global_series = False
        self.statusBar().showMessage(
            "Applied global fitted curves to the selected workspace cards. Previous per-card fits were replaced; the independent reference fits remain in the global dashboard.",
            12_000,
        )

    def _invalidate_global_series(self, reason: str) -> None:
        """Remove a project-level result after one of its scientific inputs changes."""

        if getattr(self, "_applying_global_series", False) or self._global_series_result is None:
            return
        self._global_series_result = None
        if hasattr(self, "global_report_button"):
            self.global_report_button.setEnabled(False)
            self.global_progress.setValue(0)
            self.global_progress.setFormat("Ready")
            self.global_status.setText(reason)
