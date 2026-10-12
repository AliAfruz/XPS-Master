"""Project persistence, recovery, and complete-workspace export workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..fitting.model import ParameterTie
from ..io.excel_export import ExcelExportError, export_workspace_to_excel
from ..io.project_io import WorkspaceItemData, load_project, save_project as write_project_file


class ProjectControllerMixin:
    """Persist or restore complete workspaces without owning editor widgets."""

    def _synchronize_active_workspace(self) -> None:
        """Copy the visible editor state onto its active card before persistence."""

        index = self._active_workspace_index
        if index is None or not 0 <= index < len(self._workspace_spectra):
            return
        if self.spectrum is not None:
            self._workspace_spectra[index] = self.spectrum
        self._workspace_background_settings[index] = (
            self.background_method,
            dict(self._background_parameters),
        )
        self._store_active_workspace_fit_state(refresh=False, mark_dirty=False)
        self._workspace_analyses[index] = self.survey_analysis

    def _workspace_items_for_project(self) -> tuple[WorkspaceItemData, ...]:
        self._synchronize_active_workspace()
        return tuple(
            WorkspaceItemData(
                spectrum=spectrum,
                background_method=self._workspace_background_settings[index][0],
                background_parameters=dict(self._workspace_background_settings[index][1]),
                peaks=tuple(dict(row) for row in self._workspace_peak_rows[index]),
                ties=tuple(asdict(tie) for tie in self._workspace_ties[index]),
                fit_result=self._workspace_fit_results[index],
                survey_analysis=self._workspace_analyses[index],
                consensus_result=self._workspace_consensus_results[index],
                model_comparison=self._workspace_model_comparisons[index],
            )
            for index, spectrum in enumerate(self._workspace_spectra)
        )

    def export_workspace_excel(self) -> bool:
        """Export every workspace plot, component, residual, and provenance record."""

        return self._export_workspace_excel(open_in_plotter=False)

    def export_workspace_excel_and_plot(self) -> bool:
        """Create the auditable workbook and open it in Manuscript Plotter."""

        return self._export_workspace_excel(open_in_plotter=True)

    def _export_workspace_excel(self, *, open_in_plotter: bool) -> bool:
        """Shared export path used by the normal and manuscript workflows."""

        if self.spectrum is None or not self._workspace_spectra or self._active_workspace_index is None:
            self.show_error("Nothing to export", "Load at least one spectrum before creating an Excel workbook.")
            return False
        project_stem = self._current_project_path.stem if self._current_project_path is not None else "XPS-master-Workspace"
        suggested = str(Path(self._last_directory) / f"{project_stem}-complete-analysis.xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export complete XPS workspace to Excel",
            suggested,
            "Excel workbook (*.xlsx)",
        )
        if not path:
            return False
        target = Path(path)
        if target.suffix.lower() != ".xlsx":
            target = target.with_suffix(".xlsx")
        try:
            items = self._workspace_items_for_project()
            sheet_names = export_workspace_to_excel(
                target,
                items,
                active_workspace_index=self._active_workspace_index,
                project_name=project_stem,
                global_series_result=self._global_series_result,
            )
        except (ExcelExportError, OSError, TypeError, ValueError) as error:
            self.show_error("Could not export Excel workbook", str(error))
            return False
        self._last_directory = str(target.parent)
        self.statusBar().showMessage(
            f"Exported {len(sheet_names)} spectrum sheet(s), candidate models, residuals, components, and Trust Reports to {target.name}",
            12_000,
        )
        if open_in_plotter:
            return self._launch_manuscript_plotter(target)
        return True

    def open_manuscript_plotter(self) -> bool:
        """Choose an Avantage/XPS-master workbook and open the publication editor."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Excel workbook in XPS-master Manuscript Plotter",
            self._last_directory,
            "Excel workbook (*.xlsx *.xlsm)",
        )
        if not path:
            return False
        target = Path(path)
        self._last_directory = str(target.parent)
        return self._launch_manuscript_plotter(target)

    def launch_manuscript_plotter(self) -> bool:
        """Open an empty publication editor from the integrated runner panel."""

        return self._launch_manuscript_plotter()

    def _launch_manuscript_plotter(self, workbook: Path | None = None) -> bool:
        """Start the Tk/Matplotlib publication editor outside the Qt event loop."""

        arguments = ["-m", "xps_afruz.manuscript_plotter.studio"]
        if workbook is not None:
            arguments.append(str(workbook.resolve()))
        try:
            result = QProcess.startDetached(sys.executable, arguments)
        except (OSError, RuntimeError) as error:
            self.show_error("Could not start Manuscript Plotter", str(error))
            if hasattr(self, "manuscript_runner_status"):
                self.manuscript_runner_status.setText(f"Plotter launch failed: {error}")
            return False
        started = bool(result[0]) if isinstance(result, tuple) else bool(result)
        if not started:
            self.show_error(
                "Could not start Manuscript Plotter",
                "The workbook was not changed. Run "
                "'python -m xps_afruz.manuscript_plotter.studio' from the project environment to inspect the setup.",
            )
            if hasattr(self, "manuscript_runner_status"):
                self.manuscript_runner_status.setText("The Manuscript Plotter could not be started; the workbook was not changed.")
            return False
        if workbook is None:
            message = "Launched an empty XPS-master Manuscript Plotter."
        else:
            message = f"Opened {workbook.name} in XPS-master Manuscript Plotter."
        if hasattr(self, "manuscript_runner_status"):
            self.manuscript_runner_status.setText(message)
        self.statusBar().showMessage(message, 8_000)
        return True

    def _refresh_manuscript_runner(self) -> None:
        """Keep the Publish tab honest about export readiness."""

        if not hasattr(self, "manuscript_export_button") or not hasattr(self, "manuscript_runner_status"):
            return
        ready = bool(
            self.spectrum is not None
            and self._workspace_spectra
            and self._active_workspace_index is not None
        )
        self.manuscript_export_button.setEnabled(ready)
        if not ready:
            self.manuscript_runner_status.setText(
                "No project spectra are loaded. Existing Excel workbooks and the empty plotter can still be opened."
            )
            return
        retained_fits = sum(result is not None for result in self._workspace_fit_results)
        spectrum_word = "spectrum" if len(self._workspace_spectra) == 1 else "spectra"
        fit_word = "fit" if retained_fits == 1 else "fits"
        self.manuscript_runner_status.setText(
            f"Ready: {len(self._workspace_spectra)} {spectrum_word}, {retained_fits} retained {fit_word}. "
            "Export will include unfitted spectra too."
        )

    def _project_ui_state(self) -> dict[str, object]:
        return {
            "workflow_tab": int(self.workflow_tabs.currentIndex()),
            "workspace_visible": bool(self.workspace_dock.isVisible()),
            "survey_visible": bool(self.survey_dock.isVisible()),
            "quant_visible": bool(self.quant_dock.isVisible()),
            "fit_weighting": self.fit_weighting,
            "poisson_count_scale": self.poisson_count_scale,
            "comparison_backgrounds": bool(self.comparison_backgrounds.isChecked()),
            "comparison_profiles": bool(self.comparison_profiles.isChecked()),
            "comparison_neighbors": bool(self.comparison_neighbors.isChecked()),
            "comparison_max_nfev": int(self.comparison_max_nfev.value()),
            "global_share_fwhm": bool(self.global_share_fwhm.isChecked()),
            "global_share_shape": bool(self.global_share_shape.isChecked()),
            "global_allow_shifts": bool(self.global_allow_shifts.isChecked()),
            "global_shift_limit_eV": float(self.global_shift_limit.value()),
            "global_max_nfev": int(self.global_max_nfev.value()),
        }

    def save_project(self) -> bool:
        """Save to the current project path, or use Save As for a new session."""

        if self._current_project_path is None:
            return self.save_project_as()
        return self._write_project(self._current_project_path)

    def save_project_as(self) -> bool:
        if self.spectrum is None or not self._workspace_spectra:
            self.show_error("Nothing to save", "Load a spectrum before saving a project.")
            return False
        suggested = str(self._current_project_path) if self._current_project_path is not None else self._last_directory
        path, _ = QFileDialog.getSaveFileName(self, "Save XPS project", suggested, "XPS-master project (*.xpsa)")
        if not path:
            return False
        target = Path(path)
        if target.suffix.lower() != ".xpsa":
            target = target.with_suffix(".xpsa")
        return self._write_project(target)

    def _write_project(self, path: Path, *, autosave: bool = False) -> bool:
        if self.spectrum is None or not self._workspace_spectra or self._active_workspace_index is None:
            if not autosave:
                self.show_error("Nothing to save", "Load a spectrum before saving a project.")
            return False
        try:
            items = self._workspace_items_for_project()
            active = items[self._active_workspace_index]
            write_project_file(
                path,
                spectrum=active.spectrum,
                background_method=active.background_method,
                background_parameters=active.background_parameters,
                peaks=active.peaks,
                ties=active.ties,
                analysis_settings=self._analysis_settings_for_project(),
                workspace_items=items,
                active_workspace_index=self._active_workspace_index,
                global_series_result=self._global_series_result,
                ui_state=self._project_ui_state(),
            )
        except (OSError, TypeError, ValueError) as error:
            if not autosave:
                self.show_error("Could not save project", str(error))
            return False
        if autosave:
            self.recover_project_action.setEnabled(True)
            return True
        self._current_project_path = path.resolve()
        self._last_directory = str(path.parent)
        self._remember_recent_project(self._current_project_path)
        self._discard_autosave()
        self._set_project_dirty(False)
        self.statusBar().showMessage(f"Saved complete workspace: {path.name}", 8_000)
        return True

    def _autosave_project(self) -> None:
        """Create a recovery snapshot without changing the chosen project path."""

        if not self._project_dirty or self.spectrum is None or not self._workspace_spectra:
            return
        self._write_project(self._recovery_path, autosave=True)

    def _confirm_session_replacement(self) -> bool:
        if not self._project_dirty or self.spectrum is None:
            return True
        choice = QMessageBox.warning(
            self,
            "Unsaved XPS workspace",
            "The current workspace has unsaved changes. Save them now?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return self.save_project()
        return choice == QMessageBox.StandardButton.Discard

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open XPS project", self._last_directory, "XPS-master project (*.xpsa)"
        )
        if path:
            self._open_project_path(Path(path))

    def _open_project_path(self, path: Path, *, recovered: bool = False) -> bool:
        """Replace the current session with an exact saved workspace."""

        try:
            project = load_project(path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.show_error("Could not open project", str(error))
            return False
        if not recovered and not self._confirm_session_replacement():
            return False
        try:
            ties = [[ParameterTie(**dict(tie)) for tie in item.ties] for item in project.workspace]
        except (TypeError, ValueError) as error:
            self.show_error("Could not open project", f"A saved parameter constraint is invalid: {error}")
            return False

        self._restoring_project = True
        try:
            self.spectrum = None
            self.background = None
            self.fit_result = None
            self.survey_analysis = None
            self._active_workspace_index = None
            self._workspace_spectra = [item.spectrum for item in project.workspace]
            self._workspace_background_settings = [
                (item.background_method, dict(item.background_parameters)) for item in project.workspace
            ]
            self._workspace_peak_rows = [[dict(row) for row in item.peaks] for item in project.workspace]
            self._workspace_ties = ties
            self._workspace_fit_results = [item.fit_result for item in project.workspace]
            self._workspace_analyses = [item.survey_analysis for item in project.workspace]
            self._workspace_consensus_results = [item.consensus_result for item in project.workspace]
            self._workspace_model_comparisons = [item.model_comparison for item in project.workspace]
            self._global_series_result = None
            self._refresh_workspace_views()
            self.activate_workspace_spectrum(project.active_workspace_index)
            self._restore_analysis_settings(project.analysis_settings)
            if self.survey_analysis is not None:
                self._populate_survey_results(self.survey_analysis)
                self.recalculate_quantification()
            ui = project.ui_state
            self.workflow_tabs.setCurrentIndex(
                min(max(int(ui.get("workflow_tab", 0)), 0), self.workflow_tabs.count() - 1)
            )
            weighting = str(ui.get("fit_weighting", ""))
            weighting_index = self.fit_weighting_combo.findData(weighting)
            if weighting_index >= 0:
                self.fit_weighting_combo.setCurrentIndex(weighting_index)
            count_scale = ui.get("poisson_count_scale")
            if isinstance(count_scale, (int, float)):
                self.poisson_count_scale_spin.setValue(
                    min(
                        max(float(count_scale), self.poisson_count_scale_spin.minimum()),
                        self.poisson_count_scale_spin.maximum(),
                    )
                )
            for key, widget in (
                ("comparison_backgrounds", self.comparison_backgrounds),
                ("comparison_profiles", self.comparison_profiles),
                ("comparison_neighbors", self.comparison_neighbors),
            ):
                checked = ui.get(key)
                if isinstance(checked, bool):
                    widget.setChecked(checked)
            comparison_nfev = ui.get("comparison_max_nfev")
            if isinstance(comparison_nfev, (int, float)):
                self.comparison_max_nfev.setValue(
                    min(
                        max(int(comparison_nfev), self.comparison_max_nfev.minimum()),
                        self.comparison_max_nfev.maximum(),
                    )
                )
            for key, widget in (
                ("global_share_fwhm", self.global_share_fwhm),
                ("global_share_shape", self.global_share_shape),
                ("global_allow_shifts", self.global_allow_shifts),
            ):
                checked = ui.get(key)
                if isinstance(checked, bool):
                    widget.setChecked(checked)
            shift_limit = ui.get("global_shift_limit_eV")
            if isinstance(shift_limit, (int, float)):
                self.global_shift_limit.setValue(
                    min(max(float(shift_limit), self.global_shift_limit.minimum()), self.global_shift_limit.maximum())
                )
            global_nfev = ui.get("global_max_nfev")
            if isinstance(global_nfev, (int, float)):
                self.global_max_nfev.setValue(
                    min(max(int(global_nfev), self.global_max_nfev.minimum()), self.global_max_nfev.maximum())
                )
            self._global_series_result = project.global_series_result
            if self._global_series_result is None:
                self.global_report_button.setEnabled(False)
                self.global_progress.setValue(0)
                self.global_progress.setFormat("Ready")
                self.global_status.setText(
                    "No global-series result is retained. Select related high-resolution spectra and run global fitting."
                )
            else:
                self.global_report_button.setEnabled(True)
                self.global_progress.setValue(100)
                self.global_progress.setFormat("Saved")
                self.global_status.setText(
                    f"Saved {len(self._global_series_result.spectrum_fits)}-spectrum global result. "
                    f"{self._global_series_result.preference}."
                )
            self.workspace_dock.setVisible(bool(ui.get("workspace_visible", len(project.workspace) > 1)))
            self.survey_dock.setVisible(bool(ui.get("survey_visible", True)))
            self.quant_dock.setVisible(bool(ui.get("quant_visible", True)))
            self._refresh_workspace_views()
        finally:
            self._restoring_project = False

        self._last_directory = str(path.parent)
        if recovered:
            self._current_project_path = None
            self._set_project_dirty(True)
            self.statusBar().showMessage("Recovered the autosaved workspace. Use Save As to keep it.", 10_000)
        else:
            self._current_project_path = path.resolve()
            self._remember_recent_project(self._current_project_path)
            self._discard_autosave()
            self._set_project_dirty(False)
            self.statusBar().showMessage(
                f"Opened complete workspace: {path.name} ({len(project.workspace)} spectra)", 8_000
            )
        return True

    def offer_autosave_recovery(self) -> None:
        """Offer, but never silently apply, the last recovery snapshot."""

        if not self._recovery_path.exists():
            self.recover_project_action.setEnabled(False)
            return
        choice = QMessageBox.question(
            self,
            "Recover autosaved XPS workspace",
            "An autosaved workspace is available. Recover it now?\n\n"
            "The recovered session will remain unsaved until you choose Save As.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self._open_project_path(self._recovery_path, recovered=True)

    def _analysis_settings_for_project(self) -> dict[str, object]:
        """Serialize choices and exact effective factors used by the preview."""

        factor_snapshot: list[dict[str, object]] = []
        if self.quantification is not None:
            for record in self.quantification.records:
                factor = record.sensitivity_factor
                factor_snapshot.append(
                    {
                        "element": factor.symbol,
                        "line": factor.line,
                        "value": factor.value,
                        "transmission_factor": factor.transmission_factor,
                        "analyzer_factor": factor.analyzer_factor,
                        "model": factor.model,
                        "source": factor.source,
                    }
                )
        return {
            "survey_tolerance_eV": float(self.survey_tolerance.value()),
            "selected_elements": sorted(self._survey_element_scope),
            "trace_recovery_enabled": bool(self.trace_search_checkbox.isChecked()),
            "trace_minimum_snr": float(self.trace_snr.value()),
            "excluded_elements": sorted(self._excluded_survey_elements),
            "rsf_overrides": [
                {"element": element, "line": line, "value": value}
                for (element, line), value in sorted(self._rsf_overrides.items())
            ],
            "effective_factor_snapshot": factor_snapshot,
        }

    def _restore_analysis_settings(self, settings: dict[str, Any] | Any) -> None:
        """Restore only validated, explicit screen and RSF choices."""

        if not isinstance(settings, dict):
            return
        tolerance = settings.get("survey_tolerance_eV")
        if isinstance(tolerance, (int, float)):
            self.survey_tolerance.setValue(
                min(max(float(tolerance), self.survey_tolerance.minimum()), self.survey_tolerance.maximum())
            )
        excluded = settings.get("excluded_elements", [])
        if isinstance(excluded, list):
            self._excluded_survey_elements = {str(symbol).strip() for symbol in excluded if str(symbol).strip()}
        selected = settings.get("selected_elements")
        if isinstance(selected, list):
            self._set_element_scope(
                {str(symbol).strip() for symbol in selected if str(symbol).strip()},
                invalidate=False,
            )
        trace_enabled = settings.get("trace_recovery_enabled")
        if isinstance(trace_enabled, bool):
            blocked = self.trace_search_checkbox.blockSignals(True)
            self.trace_search_checkbox.setChecked(trace_enabled)
            self.trace_search_checkbox.blockSignals(blocked)
        trace_snr = settings.get("trace_minimum_snr")
        if isinstance(trace_snr, (int, float)):
            blocked = self.trace_snr.blockSignals(True)
            self.trace_snr.setValue(min(max(float(trace_snr), self.trace_snr.minimum()), self.trace_snr.maximum()))
            self.trace_snr.blockSignals(blocked)
        overrides = settings.get("rsf_overrides", [])
        snapshot = settings.get("effective_factor_snapshot", [])
        entries = snapshot if isinstance(snapshot, list) and snapshot else overrides
        if not isinstance(entries, list):
            return
        restored: dict[tuple[str, str], float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            element, line, value = entry.get("element"), entry.get("line"), entry.get("value")
            if isinstance(element, str) and isinstance(line, str) and isinstance(value, (int, float)) and value > 0.0:
                restored[(element, line)] = float(value)
        self._rsf_overrides = restored
