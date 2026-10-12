"""Peak-table fitting and smart-deconvolution workflows."""

from __future__ import annotations

from typing import Any
from html import escape

import numpy as np
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QTableWidgetItem,
)

from ..fitting.constraint_templates import SPIN_ORBIT_TEMPLATES
from ..fitting.model import FitModel, FitResult, ParameterTie, PeakComponent, make_spin_orbit_doublet
from ..fitting.optimizer import fit_spectrum
from ..fitting.peak_shapes import PeakProfile, evaluate_peak
from ..fitting.seed_protection import diagnose_seeded_fit, protect_manual_peak_seeds
from ..fitting.smart_deconvolution import SmartDeconvolutionResult, SmartDeconvolutionSettings
from .consensus_report import open_consensus_report
from .smart_worker import SmartDeconvolutionWorker
from .widgets import NoWheelComboBox, NoWheelDoubleSpinBox


class FittingControllerMixin:
    """Manage peak components, bounded fits, and smart-search workers."""

    @staticmethod
    def _fit_metric_label(result: FitResult) -> str:
        if result.has_calibrated_chi_square:
            return f"Poisson χ²r {result.reduced_statistical_chi_square:.4g}"
        return f"RMS {result.residual_rms:.4g} (χ² uncalibrated)"

    def _fit_quality_lines(self, result: FitResult) -> str:
        """Give calibrated χ² its proper meaning and label non-statistical fits honestly."""

        unit = self.spectrum.intensity_unit if self.spectrum is not None else "intensity units"
        diagnostics = ""
        if result.objective_value is not None:
            label = "Poisson deviance" if result.objective_name == "poisson_deviance" else "Residual sum of squares"
            diagnostics += f"<br>{label}: {result.objective_value:.5g}"
        if result.jacobian_rank is not None:
            condition = "undefined" if result.jacobian_condition is None else f"{result.jacobian_condition:.4g}"
            diagnostics += f"<br>Jacobian rank: {result.jacobian_rank}; scaled condition: {condition}"
        if result.fit_warnings:
            diagnostics += "<br><b>Review:</b> " + "<br>".join(escape(warning) for warning in result.fit_warnings)
        if result.has_calibrated_chi_square:
            return (
                f"Pearson χ² (asymptotic): {result.statistical_chi_square:.5g}<br>"
                f"Reduced Pearson χ²: {result.reduced_statistical_chi_square:.5g}<br>"
                f"{result.uncertainty_model}{diagnostics}"
            )
        return (
            f"Residual RMS: {result.residual_rms:.5g} {unit}<br>"
            f"Uniform objective / DOF: {result.reduced_chi_square:.5g} (not calibrated χ²)<br>"
            f"{result.uncertainty_model}{diagnostics}"
        )

    def _remember_peak_seed_from_plot(self, event) -> None:  # type: ignore[no-untyped-def]
        """Arm an explicit energy for the next component without changing the data."""

        if self.spectrum is None or event.button() != Qt.MouseButton.LeftButton:
            return
        position = self.spectrum_plot.getPlotItem().getViewBox().mapSceneToView(event.scenePos())
        energy = float(position.x())
        lower, upper = self.spectrum.energy_range
        if not lower <= energy <= upper:
            return
        self._peak_seed_energy = energy
        self.statusBar().showMessage(f"Next + Peak will be placed at {energy:.3f} eV. Click + Peak to confirm.", 5_000)

    def _suggest_peak_seed(self) -> tuple[float, float, float, str]:
        """Choose a reproducible starting position away from already explained signal."""

        if self.spectrum is None:
            raise ValueError("No spectrum is loaded.")
        energy = self.spectrum.energy
        intensity = self.spectrum.intensity
        step = max(abs(float(np.median(np.diff(energy)))), 1e-9)
        width = max(step * 4.0, (energy[-1] - energy[0]) / 20.0)
        background = self.background if self.background is not None else np.full_like(intensity, np.median(intensity))
        rows: list[dict[str, Any]] = []
        try:
            rows = self.peak_rows()
        except ValueError:
            pass

        if self._peak_seed_energy is not None:
            selected = float(self._peak_seed_energy)
            self._peak_seed_energy = None
            index = int(np.argmin(np.abs(energy - selected)))
            height = max(float(intensity[index] - background[index]), float(np.ptp(intensity)) * 0.02, 1e-12)
            return float(energy[index]), height, width, "selected plot position"

        residual = intensity - background
        source = "largest background-corrected feature"
        if self.fit_result is not None and self.fit_result.fitted_intensity.shape == intensity.shape:
            residual = intensity - self.fit_result.fitted_intensity
            source = "largest positive fit residual"
        elif rows:
            try:
                explained = np.zeros_like(intensity)
                for component in self.components_from_table():
                    explained += evaluate_peak(
                        energy,
                        profile=component.profile,
                        center=component.center,
                        fwhm=component.fwhm,
                        area=component.area,
                        eta=component.eta,
                        asymmetry=component.asymmetry,
                        instrument_fwhm=component.instrument_fwhm,
                    )
                residual = residual - explained
                source = "largest signal not covered by existing components"
            except ValueError:
                source = "largest background-corrected feature"

        smoothing_window = int(np.clip(round(0.30 / step), 5, 25))
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        smoothed = np.convolve(residual, np.ones(smoothing_window) / smoothing_window, mode="same")
        candidates = np.maximum(smoothed, 0.0)
        allowed = np.isfinite(candidates)
        for row in rows:
            separation = max(float(row["fwhm"]) * 0.70, step * 3.0)
            allowed &= np.abs(energy - float(row["center"])) > separation
        if not np.any(allowed) or not np.any(candidates[allowed] > 0.0):
            allowed = np.isfinite(candidates)
            source = "largest remaining signal; click the intended feature to override"
        index = int(np.argmax(np.where(allowed, candidates, -np.inf)))
        height = max(float(residual[index]), float(intensity[index] - background[index]), float(np.ptp(intensity)) * 0.02, 1e-12)
        return float(energy[index]), height, width, source

    def add_peak(self) -> None:
        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum before adding a peak.")
            return
        energy = self.spectrum.energy
        peak_center, height, width, source = self._suggest_peak_seed()
        area = max(height * width * np.sqrt(np.pi / (4.0 * np.log(2.0))), float(np.ptp(self.spectrum.intensity)) * width * 1e-4, 1e-12)
        self.add_peak_row(
            PeakComponent(
                f"Peak {self.peak_table.rowCount() + 1}",
                peak_center,
                width,
                area,
                center_bounds=(float(energy[0]), float(energy[-1])),
                fwhm_bounds=(max(float(np.median(np.diff(energy))) / 2.0, 0.01), float(energy[-1] - energy[0]) * 2.0),
            )
        )
        self._invalidate_active_model_comparison("Peak components changed; rebuild the model comparison.")
        self._invalidate_global_series("Peak components changed; rerun global fitting for the selected series.")
        self.fit_result = None
        self.fit_summary.setText(
            f"Peak seeded at <b>{peak_center:.3f} eV</b> from the {source}. Review its width and line shape, then refit."
        )
        self.render_spectrum()
        self._store_active_workspace_fit_state()

    def show_peak_profile_guide(self) -> None:
        """Explain profile provenance without implying a chemical assignment."""

        guide = "\n\n".join(f"{profile.display_name}\n{profile.scientific_note}" for profile in PeakProfile)
        QMessageBox.information(
            self,
            "Peak line-shape guide",
            "Choose a profile because it is justified by the sample and acquisition model, not because it gives the "
            "smallest residual alone.\n\n"
            f"{guide}",
        )

    def add_doublet(self) -> None:
        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum before adding a doublet.")
            return
        label, accepted = QInputDialog.getText(self, "Add spin-orbit doublet", "Component label:", text="Core level")
        if not accepted or not label.strip():
            return
        template_labels = tuple(template.label for template in SPIN_ORBIT_TEMPLATES)
        template_label, accepted = QInputDialog.getItem(
            self,
            "Add spin-orbit doublet",
            "Constraint template:",
            template_labels,
            1,
            False,
        )
        if not accepted:
            return
        template = SPIN_ORBIT_TEMPLATES[template_labels.index(template_label)]
        splitting, accepted = QInputDialog.getDouble(
            self, "Add spin-orbit doublet", "Energy splitting (eV):", 1.60, 0.001, 20.0, 4
        )
        if not accepted:
            return
        ratio, accepted = QInputDialog.getDouble(
            self,
            "Add spin-orbit doublet",
            "Higher-BE / lower-BE area ratio:",
            template.upper_to_lower_area_ratio,
            0.001,
            10.0,
            4,
        )
        if not accepted:
            return
        energy = self.spectrum.energy
        center, height, width, source = self._suggest_peak_seed()
        area = max(height * width * np.sqrt(np.pi / (4.0 * np.log(2.0))), float(np.ptp(self.spectrum.intensity)) * width * 1e-4, 1e-12)
        components, ties = make_spin_orbit_doublet(
            label.strip(),
            center=center,
            fwhm=width,
            area=area,
            splitting=splitting,
            upper_to_lower_area_ratio=ratio,
            center_bounds=(float(energy[0]), float(energy[-1])),
            fwhm_bounds=(max(float(np.median(np.diff(energy))) / 2.0, 0.01), float(energy[-1] - energy[0]) * 2.0),
        )
        self._ties.extend(ties)
        for component in components:
            self.add_peak_row(component)
        self._invalidate_active_model_comparison("Peak constraints changed; rebuild the model comparison.")
        self._invalidate_global_series("Peak constraints changed; rerun global fitting for the selected series.")
        self.fit_result = None
        self.fit_summary.setText(
            f"Doublet seeded at <b>{center:.3f} eV</b> from the {source}. "
            f"Template: {template.label}. {template.note}"
        )
        self.render_spectrum()
        self._store_active_workspace_fit_state()

    def add_peak_row(self, component: PeakComponent, *, seed_center: float | None = None) -> None:
        row = self.peak_table.rowCount()
        self.peak_table.insertRow(row)
        values = (component.name, component.center, component.fwhm, component.area)
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value) if column == 0 else f"{float(value):.6g}")
            if column == 1:
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    float(component.center if seed_center is None else seed_center),
                )
            self.peak_table.setItem(row, column, item)
        profile = NoWheelComboBox()
        for option in PeakProfile:
            profile.addItem(self._profile_table_label(option), option.value)
        profile.setCurrentIndex(profile.findData(component.profile.value))
        profile.setToolTip(component.profile.scientific_note)
        profile.currentIndexChanged.connect(lambda _index, selector=profile: self._update_profile_tooltip(selector))
        self.peak_table.setCellWidget(row, 4, profile)
        self.peak_table.setItem(row, 5, QTableWidgetItem(f"{component.eta:.4g}"))
        self.peak_table.setItem(row, 6, QTableWidgetItem(f"{component.asymmetry:.4g}"))
        self.peak_table.setItem(row, 7, QTableWidgetItem(f"{component.instrument_fwhm:.4g}"))
        self.peak_table.setItem(row, 8, self._checkable_peak_item(component.vary_eta))
        self.peak_table.setItem(row, 9, self._checkable_peak_item(component.vary_asymmetry))
        self.peak_table.setItem(row, 10, self._checkable_peak_item(component.vary_instrument_fwhm))

    def _on_peak_table_item_changed(self, item: QTableWidgetItem) -> None:
        """Treat an analyst-edited center as a new persistent seed anchor."""

        if self._updating_peak_table or item.column() != 1:
            return
        try:
            value = float(item.text())
        except ValueError:
            return
        self._updating_peak_table = True
        try:
            item.setData(Qt.ItemDataRole.UserRole, value)
        finally:
            self._updating_peak_table = False

    @staticmethod
    def _checkable_peak_item(checked: bool) -> QTableWidgetItem:
        """Return one explicit shape-parameter fit toggle for a peak-table row."""

        item = QTableWidgetItem()
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def _peak_fit_toggle(self, row: int, column: int) -> bool:
        item = self.peak_table.item(row, column)
        return item is not None and item.checkState() == Qt.CheckState.Checked

    @staticmethod
    def _update_profile_tooltip(selector: QComboBox) -> None:
        """Keep the selected profile's scientific-use note available in the UI."""

        selector.setToolTip(PeakProfile(str(selector.currentData())).scientific_note)

    @staticmethod
    def _profile_table_label(profile: PeakProfile) -> str:
        """Return a compact name suitable for the narrow component inspector."""

        labels = {
            PeakProfile.GAUSSIAN: "Gaussian",
            PeakProfile.LORENTZIAN: "Lorentzian",
            PeakProfile.PSEUDO_VOIGT: "P-Voigt",
            PeakProfile.GAUSSIAN_LORENTZIAN_PRODUCT: "G-L product",
            PeakProfile.VOIGT: "Voigt",
            PeakProfile.ASYMMETRIC_PSEUDO_VOIGT: "Asym. P-Voigt",
            PeakProfile.DONIACH_SUNJIC: "DS metallic",
        }
        return labels[profile]

    def configure_selected_peak_shape(self) -> None:
        """Edit advanced shape values in a focused dialog instead of a wide table."""

        selected_rows = sorted({index.row() for index in self.peak_table.selectedIndexes()})
        if len(selected_rows) != 1:
            self.show_error("Select one component", "Select exactly one peak row before opening Shape & constraints.")
            return
        row = selected_rows[0]
        table_profile = self.peak_table.cellWidget(row, 4)
        if not isinstance(table_profile, QComboBox):
            self.show_error("Peak row incomplete", "The selected peak has no line-shape selector.")
            return

        def cell_number(column: int) -> float:
            item = self.peak_table.item(row, column)
            return float(item.text()) if item is not None else 0.0

        dialog = QDialog(self)
        dialog.setWindowTitle("Peak shape & constraints")
        dialog.setMinimumWidth(410)
        form = QFormLayout(dialog)
        form.setContentsMargins(18, 16, 18, 16)
        form.setSpacing(10)
        profile_selector = NoWheelComboBox()
        for option in PeakProfile:
            profile_selector.addItem(option.display_name, option.value)
        profile_selector.setCurrentIndex(profile_selector.findData(table_profile.currentData()))
        form.addRow("Line shape", profile_selector)

        eta = NoWheelDoubleSpinBox()
        eta.setRange(0.0, 1.0)
        eta.setDecimals(4)
        eta.setSingleStep(0.05)
        eta.setValue(cell_number(5))
        form.addRow("G/L mix η", eta)
        asymmetry = NoWheelDoubleSpinBox()
        asymmetry.setRange(-0.90, 0.90)
        asymmetry.setDecimals(4)
        asymmetry.setSingleStep(0.02)
        asymmetry.setValue(cell_number(6))
        form.addRow("Asymmetry α", asymmetry)
        instrument = NoWheelDoubleSpinBox()
        instrument.setRange(0.0, 100.0)
        instrument.setDecimals(4)
        instrument.setSingleStep(0.05)
        instrument.setValue(cell_number(7))
        form.addRow("Instrument FWHM", instrument)

        fit_eta = QCheckBox("Fit η")
        fit_eta.setChecked(self._peak_fit_toggle(row, 8))
        fit_asymmetry = QCheckBox("Fit α")
        fit_asymmetry.setChecked(self._peak_fit_toggle(row, 9))
        fit_instrument = QCheckBox("Fit instrument FWHM")
        fit_instrument.setChecked(self._peak_fit_toggle(row, 10))
        form.addRow("Optimization", fit_eta)
        form.addRow("", fit_asymmetry)
        form.addRow("", fit_instrument)
        note = QLabel()
        note.setObjectName("surveyNote")
        note.setWordWrap(True)
        form.addRow(note)

        def update_shape_controls() -> None:
            selected = PeakProfile(str(profile_selector.currentData()))
            eta_allowed = selected in {
                PeakProfile.PSEUDO_VOIGT,
                PeakProfile.VOIGT,
                PeakProfile.ASYMMETRIC_PSEUDO_VOIGT,
            }
            asymmetry_allowed = selected in {PeakProfile.ASYMMETRIC_PSEUDO_VOIGT, PeakProfile.DONIACH_SUNJIC}
            instrument_allowed = selected is PeakProfile.DONIACH_SUNJIC
            eta.setEnabled(eta_allowed)
            fit_eta.setEnabled(eta_allowed)
            asymmetry.setEnabled(asymmetry_allowed)
            fit_asymmetry.setEnabled(asymmetry_allowed)
            instrument.setEnabled(instrument_allowed)
            fit_instrument.setEnabled(instrument_allowed)
            asymmetry.setMinimum(0.0 if selected is PeakProfile.DONIACH_SUNJIC else -0.90)
            if selected is PeakProfile.DONIACH_SUNJIC and asymmetry.value() < 0.0:
                asymmetry.setValue(0.0)
            if not eta_allowed:
                fit_eta.setChecked(False)
            if not asymmetry_allowed:
                fit_asymmetry.setChecked(False)
            if not instrument_allowed:
                fit_instrument.setChecked(False)
            note.setText(selected.scientific_note)

        profile_selector.currentIndexChanged.connect(update_shape_controls)
        update_shape_controls()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._invalidate_active_model_comparison("Peak line shape changed; rebuild the model comparison.")
        self._invalidate_global_series("Peak line shape changed; rerun global fitting for the selected series.")
        table_profile.setCurrentIndex(table_profile.findData(profile_selector.currentData()))
        for column, value in ((5, eta.value()), (6, asymmetry.value()), (7, instrument.value())):
            item = self.peak_table.item(row, column)
            if item is not None:
                item.setText(f"{value:.6g}")
        for column, checked in ((8, fit_eta.isChecked()), (9, fit_asymmetry.isChecked()), (10, fit_instrument.isChecked())):
            item = self.peak_table.item(row, column)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.fit_result = None
        self.render_spectrum()
        self._store_active_workspace_fit_state()

    def remove_selected_peaks(self) -> None:
        selected_rows = sorted({index.row() for index in self.peak_table.selectedIndexes()}, reverse=True)
        if selected_rows:
            self._invalidate_active_model_comparison("Peak components changed; rebuild the model comparison.")
            self._invalidate_global_series("Peak components changed; rerun global fitting for the selected series.")
        for row in selected_rows:
            self.peak_table.removeRow(row)
        self._prune_invalid_ties()
        self.fit_result = None
        self.render_spectrum()
        self._store_active_workspace_fit_state()

    def peak_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.peak_table.rowCount()):
            profile_widget = self.peak_table.cellWidget(row, 4)
            if not isinstance(profile_widget, QComboBox):
                raise ValueError(f"Peak row {row + 1} has no profile selector.")
            values = [self.peak_table.item(row, column) for column in (0, 1, 2, 3, 5, 6, 7)]
            if any(value is None for value in values):
                raise ValueError(f"Peak row {row + 1} is incomplete.")
            try:
                seed_value = self.peak_table.item(row, 1).data(Qt.ItemDataRole.UserRole)
                rows.append(
                    {
                        "name": values[0].text().strip(),
                        "center": float(values[1].text()),
                        "seed_center": float(values[1].text()) if seed_value is None else float(seed_value),
                        "fwhm": float(values[2].text()),
                        "area": float(values[3].text()),
                        "profile": str(profile_widget.currentData()),
                        "eta": float(values[4].text()),
                        "asymmetry": float(values[5].text()),
                        "instrument_fwhm": float(values[6].text()),
                        "vary_eta": self._peak_fit_toggle(row, 8),
                        "vary_asymmetry": self._peak_fit_toggle(row, 9),
                        "vary_instrument_fwhm": self._peak_fit_toggle(row, 10),
                    }
                )
            except ValueError as error:
                raise ValueError(f"Peak row {row + 1} contains an invalid number.") from error
        return rows

    def components_from_table(self) -> list[PeakComponent]:
        if self.spectrum is None:
            raise ValueError("No spectrum is loaded.")
        energy = self.spectrum.energy
        step = float(np.median(np.diff(energy)))
        components: list[PeakComponent] = []
        rows = self.peak_rows()
        for row in rows:
            if not row["name"]:
                raise ValueError("Every peak needs a name.")
            profile = PeakProfile(str(row["profile"]))
            eta_profiles = {PeakProfile.PSEUDO_VOIGT, PeakProfile.VOIGT, PeakProfile.ASYMMETRIC_PSEUDO_VOIGT}
            asymmetry_profiles = {PeakProfile.ASYMMETRIC_PSEUDO_VOIGT, PeakProfile.DONIACH_SUNJIC}
            if row["vary_eta"] and profile not in eta_profiles:
                raise ValueError(f"Fit eta is not applicable to {profile.display_name}.")
            if row["vary_asymmetry"] and profile not in asymmetry_profiles:
                raise ValueError(f"Fit alpha is not applicable to {profile.display_name}.")
            if row["vary_instrument_fwhm"] and profile is not PeakProfile.DONIACH_SUNJIC:
                raise ValueError("Fit instrument FWHM is available only for Doniach-Sunjic.")
            components.append(
                PeakComponent(
                    row["name"],
                    center=row["center"],
                    fwhm=row["fwhm"],
                    area=row["area"],
                    profile=profile,
                    eta=row["eta"],
                    asymmetry=row["asymmetry"],
                    instrument_fwhm=row["instrument_fwhm"],
                    center_bounds=(float(energy[0]), float(energy[-1])),
                    fwhm_bounds=(max(step / 2.0, 0.01), float(energy[-1] - energy[0]) * 2.0),
                    asymmetry_bounds=(0.0, 0.90) if profile is PeakProfile.DONIACH_SUNJIC else (-0.90, 0.90),
                    instrument_fwhm_bounds=(0.0, max(20.0, float(energy[-1] - energy[0]) * 2.0)),
                    vary_eta=bool(row["vary_eta"]),
                    vary_asymmetry=bool(row["vary_asymmetry"]),
                    vary_instrument_fwhm=bool(row["vary_instrument_fwhm"]),
                )
            )
        if not components:
            raise ValueError("Add at least one peak component before fitting.")
        if getattr(self, "seed_guardrails_checkbox", None) is not None and self.seed_guardrails_checkbox.isChecked():
            components = list(
                protect_manual_peak_seeds(
                    energy,
                    components,
                    self.active_ties(),
                    seed_centers={str(row["name"]): float(row["seed_center"]) for row in rows},
                )
            )
        return components

    def active_ties(self) -> list[ParameterTie]:
        parameter_names = {
            f"{row['name']}.{field}"
            for row in self.peak_rows()
            for field in ("center", "fwhm", "area", "eta", "asymmetry", "instrument_fwhm")
        }
        return [tie for tie in self._ties if tie.target in parameter_names and tie.source in parameter_names]

    def _prune_invalid_ties(self) -> None:
        try:
            self._ties = self.active_ties()
        except ValueError:
            self._ties = []

    def run_fit(self) -> None:
        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum before fitting.")
            return
        if self.background is None:
            self.recalculate_background()
        try:
            components = self.components_from_table()
            ties = self.active_ties()
            model = FitModel(components, self.background, ties, name=self.spectrum.label)
            result = fit_spectrum(
                self.spectrum,
                model,
                weighting=self.fit_weighting,  # type: ignore[arg-type]
                count_scale=self.poisson_count_scale,
            )
            protection_enabled = self.seed_guardrails_checkbox.isChecked()
            protection_diagnostics = (
                diagnose_seeded_fit(result, components, self.spectrum.energy) if protection_enabled else None
            )
        except ValueError as error:
            self.show_error("Fit could not start", str(error))
            return
        self.fit_result = result
        self._invalidate_active_model_comparison("The active fit parameters changed; rebuild the model comparison.")
        self._invalidate_global_series("The active peak parameters changed; rerun global fitting for the selected series.")
        self._update_table_values(result.parameter_values)
        self.render_spectrum()
        tie_text = f", {len(ties)} constraint(s)" if ties else ""
        if protection_diagnostics is None:
            protection_text = (
                "<b>Smart seed protection: OFF</b> — centers may cross and widths may expand across the ROI."
            )
        else:
            review: list[str] = []
            if protection_diagnostics.collapsed_pairs:
                names = ", ".join(f"{left}/{right}" for left, right in protection_diagnostics.collapsed_pairs)
                review.append(f"unresolved strong overlap: {names}")
            if protection_diagnostics.weak_components:
                review.append("negligible fitted area: " + ", ".join(protection_diagnostics.weak_components))
            if protection_diagnostics.center_boundary_components:
                review.append(
                    "center reached a local-zone edge: "
                    + ", ".join(protection_diagnostics.center_boundary_components)
                )
            if protection_diagnostics.width_boundary_components:
                review.append(
                    "width reached its local ceiling: "
                    + ", ".join(protection_diagnostics.width_boundary_components)
                )
            if abs(protection_diagnostics.residual_autocorrelation) >= 0.50:
                review.append(
                    f"structured residual (lag-1 r={protection_diagnostics.residual_autocorrelation:.2f})"
                )
            protection_text = (
                "<b>Smart seed protection: ON</b> — ordered local center zones and finite width limits.<br>"
                + ("Review: " + "; ".join(review) if review else "Separation check: no numerical peak pile-up detected.")
            )
        self.fit_summary.setText(
            f"<b>{'Fit converged' if result.success else 'Fit stopped'}</b>{tie_text}<br>"
            f"Background: {self._background_description()}<br>"
            f"{self._fit_quality_lines(result)}<br>"
            f"{protection_text}<br>"
            f"Degrees of freedom: {result.degrees_of_freedom}<br>"
            f"{result.message}"
        )
        self._store_active_workspace_fit_state()
        if protection_diagnostics is not None and protection_diagnostics.review_needed:
            self.statusBar().showMessage(
                "Fit complete with seed protection; review the flagged overlap, boundary, or residual warning.",
                10_000,
            )
        else:
            self.statusBar().showMessage("Fit complete." if result.success else "Fit stopped before convergence.")

    def start_smart_deconvolution(self) -> None:
        """Launch a responsive, residual-aware multi-start component search."""

        if self._smart_thread is not None and self._smart_thread.isRunning():
            self.statusBar().showMessage("Smart deconvolution is already running.")
            return
        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            self.show_error("Global fitting is running", "Finish or stop global multi-spectrum fitting first.")
            return
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            self.show_error("Model comparison is running", "Finish or stop model comparison before smart deconvolution.")
            return
        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a high-resolution spectrum before starting smart deconvolution.")
            return
        if self.background is None:
            self.recalculate_background()
        if self.background is None:
            self.show_error("No background", "Choose a valid background before starting smart deconvolution.")
            return
        try:
            settings = SmartDeconvolutionSettings(
                peak_count=int(self.smart_peak_count.value()),
                profile=str(self.smart_profile.currentData()),
                candidate_trials=int(self.smart_trials.value()),
                weighting=self.fit_weighting,
                count_scale=self.poisson_count_scale,
            )
        except ValueError as error:
            self.show_error("Smart deconvolution settings", str(error))
            return

        self._smart_workspace_index = self._active_workspace_index
        self._smart_target_spectrum = self.spectrum
        thread = QThread(self)
        worker = SmartDeconvolutionWorker(self.spectrum, self.background, settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_smart_deconvolution_progress)
        worker.completed.connect(self._apply_smart_deconvolution_result)
        worker.failed.connect(self._smart_deconvolution_failed)
        worker.cancelled.connect(self._smart_deconvolution_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._smart_deconvolution_finished)
        self._smart_thread = thread
        self._smart_worker = worker
        self._set_smart_deconvolution_running(True)
        self.smart_progress.setValue(0)
        self.smart_progress.setFormat("0%")
        self.smart_status.setText(
            f"Preparing {settings.candidate_trials} bounded {settings.peak_count}-component models in a background worker…"
        )
        self.statusBar().showMessage("Smart deconvolution started. The interface remains available while candidates are evaluated.")
        thread.start()

    def cancel_smart_deconvolution(self) -> None:
        """Ask the worker to stop safely after its current bounded candidate fit."""

        if self._smart_worker is None:
            return
        self._smart_worker.cancel()
        self.smart_cancel_button.setEnabled(False)
        self.smart_status.setText("Stopping after the current candidate fit finishes…")

    def _set_smart_deconvolution_running(self, running: bool) -> None:
        self.smart_start_button.setEnabled(not running)
        self.smart_cancel_button.setEnabled(running)
        self.smart_peak_count.setEnabled(not running)
        self.smart_profile.setEnabled(not running)
        self.smart_trials.setEnabled(not running)
        if hasattr(self, "consensus_start_button"):
            consensus_running = self._consensus_thread is not None and self._consensus_thread.isRunning()
            self.consensus_start_button.setEnabled(not running and not consensus_running)
        if hasattr(self, "comparison_start_button"):
            comparison_running = self._comparison_thread is not None and self._comparison_thread.isRunning()
            self.comparison_start_button.setEnabled(not running and not comparison_running)

    def _update_smart_deconvolution_progress(self, percentage: int, message: str) -> None:
        self.smart_progress.setValue(percentage)
        self.smart_progress.setFormat(f"{percentage}%")
        self.smart_status.setText(message)

    def _apply_smart_deconvolution_result(self, result: SmartDeconvolutionResult) -> None:
        """Apply only to the spectrum that was active when the search began."""

        if self.spectrum is not self._smart_target_spectrum or self._active_workspace_index != self._smart_workspace_index:
            self.smart_status.setText("Search completed, but its spectrum was changed before completion; the result was not applied.")
            self.statusBar().showMessage("Smart-deconvolution result was kept out of the wrong workspace card.", 8_000)
            return
        self.peak_table.setRowCount(0)
        self._ties = []
        self._invalidate_active_model_comparison("Smart deconvolution changed the fit hypothesis; rebuild model comparison.")
        self._invalidate_global_series("Smart deconvolution changed the peak hypothesis; rerun global fitting.")
        for component in result.components:
            self.add_peak_row(component)
        self.fit_result = result.fit_result
        self._update_table_values(result.fit_result.parameter_values)
        self.render_spectrum()
        self.fit_summary.setText(
            f"<b>Smart deconvolution complete</b><br>"
            f"{len(result.components)} requested component(s) • {result.completed_trials}/{result.candidate_trials} bounded candidate fits<br>"
            f"{self._fit_quality_lines(result.fit_result)}<br>"
            f"Model score: {result.score:.5g}<br>"
            f"BIC: {result.bic:.5g} • residual autocorrelation: {result.residual_autocorrelation:.3f}<br>"
            f"Background: {self._background_description()}"
        )
        self.smart_progress.setValue(100)
        self.smart_progress.setFormat("100%")
        self.smart_status.setText(
            "Best candidate applied. The score compares numerical models; it is not a chemical-state assignment. Review all components."
        )
        self._store_active_workspace_fit_state()
        self.statusBar().showMessage("Smart deconvolution complete. Review components, background, and residual before reporting.", 10_000)

    def _smart_deconvolution_failed(self, message: str) -> None:
        self.smart_progress.setValue(0)
        self.smart_progress.setFormat("Not completed")
        self.smart_status.setText(f"Smart deconvolution could not complete: {message}")
        self.statusBar().showMessage("Smart deconvolution did not complete.", 8_000)

    def _smart_deconvolution_cancelled(self) -> None:
        self.smart_progress.setFormat("Stopped")
        self.smart_status.setText("Smart deconvolution stopped. The existing fit was left unchanged.")
        self.statusBar().showMessage("Smart deconvolution stopped; no partial model was applied.", 8_000)

    def _smart_deconvolution_finished(self) -> None:
        self._smart_thread = None
        self._smart_worker = None
        self._set_smart_deconvolution_running(False)

    def show_stability_consensus_report(self) -> None:
        """Open the presentation-only report for the active retained evidence."""

        result = self._consensus_result
        if result is None:
            self.show_error(
                "No Stability Trust Report",
                "Run Stability-Consensus on the active spectrum first.",
            )
            return
        open_consensus_report(self, result, install_table_menu=self._install_table_copy_menu)

    def _update_table_values(self, values: dict[str, float] | Any) -> None:
        self._updating_peak_table = True
        try:
            for row in range(self.peak_table.rowCount()):
                name_item = self.peak_table.item(row, 0)
                if name_item is None:
                    continue
                name = name_item.text()
                for column, field in (
                    (1, "center"),
                    (2, "fwhm"),
                    (3, "area"),
                    (5, "eta"),
                    (6, "asymmetry"),
                    (7, "instrument_fwhm"),
                ):
                    item = self.peak_table.item(row, column)
                    key = f"{name}.{field}"
                    if item is not None and key in values:
                        item.setText(f"{float(values[key]):.6g}")
        finally:
            self._updating_peak_table = False

    def clear_fit(self) -> None:
        self.fit_result = None
        self.fit_summary.setText("No fit has been run.")
        self.render_spectrum()
        self._store_active_workspace_fit_state()
