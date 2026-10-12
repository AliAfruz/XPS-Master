"""Energy-axis conversion and calibration workflows for the main window."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from ..chemistry.survey import SurveyAnalysis, analyze_survey
from ..core.energy import (
    EnergyCalibration,
    EnergyScale,
    apply_energy_calibration,
    calibration_comparison,
    charge_correct,
    convert_energy_scale,
)
from ..core.spectrum import Spectrum
from ..fitting.model import shift_fit_result_energy, shift_parameter_ties_energy
from ..fitting.model_comparison import shift_model_comparison_energy
from ..fitting.global_series import shift_global_series_energy
from .widgets import NoWheelDoubleSpinBox


class EnergyControllerMixin:
    """Coordinate active and project-wide energy transformations."""

    def _replace_energy_transformed_spectrum(self, spectrum: Spectrum, *, operation: str) -> None:
        """Apply an energy transform while invalidating position-dependent fits."""

        was_screened = self.survey_analysis is not None
        self.spectrum = spectrum
        self._store_active_workspace_spectrum(spectrum)
        self.fit_result = None
        self.peak_table.setRowCount(0)
        self._ties = []
        self._reset_survey_results(reset_preferences=False)
        self.setWindowTitle(f"XPS-master — {spectrum.label}")
        self.recalculate_background()
        if was_screened:
            self.analyze_survey()
        self.statusBar().showMessage(operation)

    def convert_active_energy(self, target_scale: EnergyScale) -> None:
        """Convert the active spectrum only when its photon energy is explicit."""

        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum before converting its energy axis.")
            return
        try:
            converted = convert_energy_scale(self.spectrum, target_scale)
        except ValueError as error:
            self.show_error("Energy conversion failed", str(error))
            return
        if converted is self.spectrum:
            self.statusBar().showMessage(f"Spectrum is already on the {target_scale.value}-energy scale.")
            return
        self._replace_energy_transformed_spectrum(
            converted,
            operation=f"Converted active axis to {target_scale.value} energy.",
        )

    def charge_correct_spectrum(self) -> None:
        """Collect an analyst-selected reference and apply a documented charge shift."""

        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum before applying charge correction.")
            return
        if self.spectrum.energy_scale is not EnergyScale.BINDING:
            self.show_error("Binding-energy required", "Convert the spectrum to binding energy before charge correction.")
            return
        default_observed = (
            self._survey_table_matches[self.survey_table.currentRow()].peak.maximum_energy
            if 0 <= self.survey_table.currentRow() < len(self._survey_table_matches)
            else float(self.spectrum.energy[np.argmax(self.spectrum.intensity)])
        )
        label, accepted = QInputDialog.getText(self, "Charge correction", "Reference label:", text="C 1s")
        if not accepted or not label.strip():
            return
        observed, accepted = QInputDialog.getDouble(
            self, "Charge correction", "Observed reference position (eV):", default_observed, -100.0, 2_000.0, 4
        )
        if not accepted:
            return
        reference, accepted = QInputDialog.getDouble(
            self, "Charge correction", "Target reference position (eV):", 284.8, -100.0, 2_000.0, 4
        )
        if not accepted:
            return
        uncertainty, accepted = QInputDialog.getDouble(
            self, "Charge correction", "Reference uncertainty (eV):", 0.20, 0.0, 20.0, 4
        )
        if not accepted:
            return
        try:
            corrected = charge_correct(
                self.spectrum,
                observed_energy_eV=observed,
                reference_energy_eV=reference,
                reference_label=label.strip(),
                uncertainty_eV=uncertainty,
            )
        except ValueError as error:
            self.show_error("Charge correction failed", str(error))
            return
        self._replace_energy_transformed_spectrum(
            corrected,
            operation=f"Applied {reference - observed:+.3f} eV charge correction using {label.strip()}.",
        )

    def calibrate_project_energy(self) -> None:
        """Collect one signed coordinate shift and apply it to every workspace spectrum."""

        if not self._workspace_spectra:
            self.show_error("No project spectra", "Load one or more spectra before applying a project-wide shift.")
            return
        if self._smart_thread is not None and self._smart_thread.isRunning():
            self.show_error(
                "Deconvolution is running",
                "Stop or finish smart deconvolution before changing every project energy axis.",
            )
            return
        if self._consensus_thread is not None and self._consensus_thread.isRunning():
            self.show_error(
                "Consensus analysis is running",
                "Stop or finish stability consensus before changing every project energy axis.",
            )
            return
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            self.show_error(
                "Model comparison is running",
                "Stop or finish model comparison before changing every project energy axis.",
            )
            return
        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            self.show_error(
                "Global fitting is running",
                "Stop or finish global multi-spectrum fitting before changing every project energy axis.",
            )
            return

        scale_names = sorted({spectrum.energy_scale.value for spectrum in self._workspace_spectra})
        dialog = QDialog(self)
        dialog.setWindowTitle("Project-wide energy calibration")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(13)

        title = QLabel("APPLY ONE SIGNED ENERGY SHIFT TO THE COMPLETE PROJECT")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        note = QLabel(
            f"This will add the same signed coordinate offset to all {len(self._workspace_spectra)} loaded "
            f"spectra ({', '.join(scale_names)} scale). Imported axes remain preserved. Retained fit centers and "
            "constraints move with the axes; fitted intensities and residuals remain unchanged. Existing survey "
            "screens are recalculated."
        )
        note.setObjectName("surveyNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        shift_input = NoWheelDoubleSpinBox()
        shift_input.setRange(-100.0, 100.0)
        shift_input.setDecimals(5)
        shift_input.setSingleStep(0.05)
        shift_input.setSuffix(" eV")
        shift_input.setToolTip("Positive values move every active energy coordinate to higher displayed energy.")
        reference_input = QLineEdit("Project-wide manual calibration")
        reference_input.setPlaceholderText("Reference or reason for calibration")
        uncertainty_input = NoWheelDoubleSpinBox()
        uncertainty_input.setRange(0.0, 20.0)
        uncertainty_input.setDecimals(5)
        uncertainty_input.setSingleStep(0.01)
        uncertainty_input.setSuffix(" eV")
        uncertainty_input.setValue(0.0)
        form.addRow("Signed shift:", shift_input)
        form.addRow("Reference / reason:", reference_input)
        form.addRow("Shift uncertainty:", uncertainty_input)
        layout.addLayout(form)

        direction = QLabel("Example: +1.200 eV changes a peak at 284.800 eV to 286.000 eV on every active axis.")
        direction.setObjectName("surveyNote")
        direction.setWordWrap(True)
        layout.addWidget(direction)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if apply_button is not None:
            apply_button.setText(f"Apply to {len(self._workspace_spectra)} spectra")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not reference_input.text().strip():
            self.show_error("Calibration reference required", "Enter a reference or reason for this project-wide shift.")
            return
        shift = float(shift_input.value())
        if np.isclose(shift, 0.0, rtol=0.0, atol=1e-12):
            self.statusBar().showMessage("No project calibration was applied because the selected shift is 0 eV.")
            return
        try:
            self.apply_project_energy_shift(
                shift,
                reference_label=reference_input.text().strip(),
                uncertainty_eV=float(uncertainty_input.value()),
            )
        except (RuntimeError, ValueError) as error:
            self.show_error("Project calibration failed", str(error))

    def apply_project_energy_shift(
        self,
        shift_eV: float,
        *,
        reference_label: str,
        uncertainty_eV: float = 0.0,
    ) -> int:
        """Apply an auditable uniform shift to every spectrum and retained result."""

        if self._smart_thread is not None and self._smart_thread.isRunning():
            raise RuntimeError("Smart deconvolution must finish or be stopped before project calibration.")
        if self._consensus_thread is not None and self._consensus_thread.isRunning():
            raise RuntimeError("Stability consensus must finish or be stopped before project calibration.")
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            raise RuntimeError("Model comparison must finish or be stopped before project calibration.")
        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            raise RuntimeError("Global multi-spectrum fitting must finish or be stopped before project calibration.")
        if not self._workspace_spectra or self._active_workspace_index is None:
            raise ValueError("The project has no spectra to calibrate.")
        shift = float(shift_eV)
        uncertainty = float(uncertainty_eV)
        if not np.isfinite(shift):
            raise ValueError("Project energy shift must be finite.")
        if uncertainty < 0.0 or not np.isfinite(uncertainty):
            raise ValueError("Project calibration uncertainty must be finite and non-negative.")
        if not reference_label.strip():
            raise ValueError("A project calibration reference or reason is required.")

        self._synchronize_active_workspace()
        count = len(self._workspace_spectra)
        calibration = EnergyCalibration(
            shift_eV=shift,
            reference_label=reference_label.strip(),
            uncertainty_eV=uncertainty,
            method="project_wide_manual_shift",
            scope="complete_workspace",
            affected_spectra=count,
        )
        shifted_spectra = [apply_energy_calibration(spectrum, calibration) for spectrum in self._workspace_spectra]
        shifted_rows = [
            [
                {
                    **row,
                    "center": float(row["center"]) + shift,
                    "seed_center": float(row.get("seed_center", row["center"])) + shift,
                }
                for row in rows
            ]
            for rows in self._workspace_peak_rows
        ]
        shifted_ties = [shift_parameter_ties_energy(ties, shift) for ties in self._workspace_ties]
        shifted_results = [
            None if result is None else shift_fit_result_energy(result, shift)
            for result in self._workspace_fit_results
        ]
        shifted_comparisons = [
            None if result is None else shift_model_comparison_energy(result, shift)
            for result in self._workspace_model_comparisons
        ]
        shifted_global_series = (
            None
            if self._global_series_result is None
            else shift_global_series_energy(self._global_series_result, shift)
        )

        refreshed_analyses: list[SurveyAnalysis | None] = []
        for spectrum, analysis in zip(shifted_spectra, self._workspace_analyses, strict=True):
            if analysis is None:
                refreshed_analyses.append(None)
                continue
            try:
                refreshed_analyses.append(
                    analyze_survey(
                        spectrum,
                        tolerance=analysis.tolerance,
                        include_elements=analysis.included_elements,
                        exclude_elements=analysis.excluded_elements,
                        target_elements=analysis.trace_searched_elements,
                        trace_minimum_snr=analysis.trace_minimum_snr,
                    )
                )
            except ValueError:
                refreshed_analyses.append(None)

        self._workspace_spectra = shifted_spectra
        self._workspace_peak_rows = shifted_rows
        self._workspace_ties = shifted_ties
        self._workspace_fit_results = shifted_results
        self._workspace_model_comparisons = shifted_comparisons
        self._global_series_result = shifted_global_series
        self._workspace_analyses = refreshed_analyses
        self._workspace_consensus_results = [None] * count
        self._consensus_result = None

        active = self._active_workspace_index
        self.spectrum = self._workspace_spectra[active]
        self._spectrum_undo.clear()
        self._reset_survey_results(reset_preferences=False)
        self.recalculate_background()
        self._restore_workspace_fit_state(active)
        self._restore_workspace_model_comparison_state(active)
        active_analysis = self._workspace_analyses[active]
        if active_analysis is not None:
            self.survey_analysis = active_analysis
            self._populate_survey_results(active_analysis)
            self.recalculate_quantification()
        self._refresh_workspace_views()
        self.render_spectrum()
        self._set_project_dirty()
        self.statusBar().showMessage(
            f"Applied {shift:+.5f} eV to all {count} project spectra; original axes and provenance were preserved.",
            10_000,
        )
        return count

    def show_calibration_provenance(self) -> None:
        """Show active/original scale comparison and every recorded operation."""

        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum to inspect its energy provenance.")
            return
        comparison = calibration_comparison(self.spectrum)
        history = self.spectrum.correction_history
        operation_lines: list[str] = []
        for entry in history:
            detail = str(entry.get("reference_label", entry.get("from_scale", ""))).strip()
            if entry.get("shift_eV") is not None:
                detail = f"{detail} · shift {float(entry['shift_eV']):+.5g} eV".strip(" ·")
            if entry.get("scope") == "complete_workspace":
                affected = entry.get("affected_spectra")
                count_text = "" if affected is None else f" ({affected} spectra)"
                detail += f" · complete workspace{count_text}"
            operation_lines.append(f"• {entry.get('operation', 'operation')}: {detail}".rstrip())
        operations = "\n".join(operation_lines) or "No conversion or calibration has been applied."
        photon = "not recorded" if self.spectrum.photon_energy_eV is None else f"{self.spectrum.photon_energy_eV:.3f} eV"
        QMessageBox.information(
            self,
            "Energy calibration provenance",
            f"Active scale: {comparison['active_scale']}\n"
            f"Imported scale: {comparison['original_scale']}\n"
            f"Photon energy: {photon}\n"
            f"Median active/original offset: {comparison['median_shift_eV']:.4g} eV\n"
            f"Tracked uncertainty: {comparison['uncertainty_eV']:.4g} eV\n\n"
            f"History\n{operations}",
        )
