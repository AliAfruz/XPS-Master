"""Main-window workflow for survey screening and XPS region fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, QStandardPaths, QThread, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..backgrounds import BackgroundDiagnostics, BackgroundMethod
from ..chemistry.quantification import CompositionResult
from ..chemistry.survey import SurveyAnalysis, SurveyMatch
from ..chemistry.survey_library import REFERENCE_LIBRARY
from ..core.energy import EnergyScale, original_axis_in_active_scale
from ..core.spectrum import Spectrum
from ..metadata import AUTHOR_NAME, AUTHOR_EMAIL, AUTHOR_WEBSITE
from ..examples.fit_synthetic import synthetic_doublet
from ..fitting.model import FitResult, ParameterTie, make_spin_orbit_doublet
from ..fitting.model_comparison import ModelComparisonResult
from ..fitting.peak_shapes import PeakProfile
from ..fitting.stability_consensus import StabilityConsensusResult
from ..fitting.global_series import GlobalSeriesResult
from ..io.delimited import read_delimited_spectrum
from ..io.vgd import VGDReadError, read_vgd_spectra
from .background_controller import BackgroundControllerMixin
from .consensus_controller import ConsensusControllerMixin
from .energy_controller import EnergyControllerMixin
from .fitting_controller import FittingControllerMixin
from .global_series_controller import GlobalSeriesControllerMixin
from .global_series_worker import GlobalSeriesWorker
from .interface_builder import InterfaceBuilderMixin
from .model_comparison_controller import ModelComparisonControllerMixin
from .model_comparison_worker import ModelComparisonWorker
from .processing_controller import ProcessingControllerMixin
from .project_controller import ProjectControllerMixin
from .smart_worker import SmartDeconvolutionWorker
from .survey_controller import SurveyControllerMixin
from .theme import SPACE_STYLESHEET
from .workspace_controller import WorkspaceControllerMixin


class MainWindow(
    InterfaceBuilderMixin,
    BackgroundControllerMixin,
    EnergyControllerMixin,
    FittingControllerMixin,
    GlobalSeriesControllerMixin,
    ModelComparisonControllerMixin,
    ProjectControllerMixin,
    SurveyControllerMixin,
    ProcessingControllerMixin,
    WorkspaceControllerMixin,
    ConsensusControllerMixin,
    QMainWindow,
):
    """An interactive, transparent fitting workspace for one XPS region."""

    PEAK_COLUMNS = (
        "Name",
        "Center (eV)",
        "FWHM (eV)",
        "Area",
        "Profile",
        "η",
        "α",
        "Inst. FWHM",
        "Fit η",
        "Fit α",
        "Fit Inst.",
    )
    SURVEY_COLUMNS = ("Element", "Feature", "Reference", "Detected", "ΔE", "SNR", "Rank")
    QUANT_COLUMNS = ("Element", "Line", "Area", "RSF", "Corrected", "at.% ±")
    TRACE_TARGET_LIMIT = 12

    def __init__(self) -> None:
        super().__init__()
        self.spectrum: Spectrum | None = None
        self.background: np.ndarray | None = None
        self.background_diagnostics: BackgroundDiagnostics | None = None
        self._background_parameters: dict[str, object] = self._default_background_parameters(BackgroundMethod.LINEAR)
        self.fit_result = None
        self.survey_analysis: SurveyAnalysis | None = None
        self._survey_table_matches: list[SurveyMatch] = []
        self._excluded_survey_elements: set[str] = set()
        self._survey_element_scope: set[str] = {reference.symbol for reference in REFERENCE_LIBRARY}
        self._element_scope_items: dict[str, QTableWidgetItem] = {}
        self._updating_element_scope = False
        self._rsf_overrides: dict[tuple[str, str], float] = {}
        self.quantification: CompositionResult | None = None
        self._updating_quant_table = False
        self._ties: list[ParameterTie] = []
        self._peak_seed_energy: float | None = None
        self._updating_peak_table = False
        self._smart_thread: QThread | None = None
        self._smart_worker: SmartDeconvolutionWorker | None = None
        self._smart_workspace_index: int | None = None
        self._smart_target_spectrum: Spectrum | None = None
        self._consensus_thread: QThread | None = None
        self._consensus_worker: StabilityConsensusWorker | None = None
        self._consensus_workspace_index: int | None = None
        self._consensus_target_spectrum: Spectrum | None = None
        self._consensus_result: StabilityConsensusResult | None = None
        self._comparison_thread: QThread | None = None
        self._comparison_worker: ModelComparisonWorker | None = None
        self._comparison_workspace_index: int | None = None
        self._comparison_target_spectrum: Spectrum | None = None
        self._model_comparison_result: ModelComparisonResult | None = None
        self._applying_model_comparison = False
        self._global_series_thread: QThread | None = None
        self._global_series_worker: GlobalSeriesWorker | None = None
        self._global_series_input_spectra: tuple[Spectrum, ...] = ()
        self._global_series_result: GlobalSeriesResult | None = None
        self._applying_global_series = False
        self._spectrum_undo: list[Spectrum] = []
        self._workspace_spectra: list[Spectrum] = []
        self._workspace_analyses: list[SurveyAnalysis | None] = []
        self._workspace_background_settings: list[tuple[BackgroundMethod, dict[str, object]]] = []
        self._workspace_fit_results: list[FitResult | None] = []
        self._workspace_peak_rows: list[list[dict[str, Any]]] = []
        self._workspace_ties: list[list[ParameterTie]] = []
        self._workspace_consensus_results: list[StabilityConsensusResult | None] = []
        self._workspace_model_comparisons: list[ModelComparisonResult | None] = []
        self._active_workspace_index: int | None = None
        self._updating_workspace_selection = False
        self._last_directory = str(Path.cwd())
        self._current_project_path: Path | None = None
        self._project_dirty = False
        self._restoring_project = False
        self._settings = QSettings("Afruz Studio", "XPS Afruz Studio")  # Retain the existing preferences namespace.
        recovery_root = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        )
        self._recovery_path = recovery_root / "recovery" / "workspace-autosave.xpsa"
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)
        self._autosave_timer.timeout.connect(self._autosave_project)
        self._autosave_timer.start()

        self._update_window_title()
        self.setMinimumSize(1_120, 720)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1_780, int(available.width() * 0.94)), min(1_020, int(available.height() * 0.92)))
        else:
            self.resize(1_680, 960)
        self.setDockOptions(
            QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.GroupedDragging
        )
        self.setStyleSheet(SPACE_STYLESHEET)
        self._build_actions()
        self._build_interface()
        self.statusBar().showMessage("Load a spectrum or open the synthetic example to begin.")

    def _update_window_title(self) -> None:
        """Show the active spectrum, project name, and unsaved state consistently."""

        context = self._current_project_path.name if self._current_project_path is not None else "Untitled project"
        if self.spectrum is not None:
            context = f"{context} — {self.spectrum.label}"
        marker = " *" if self._project_dirty else ""
        self.setWindowTitle(f"XPS-master — {context}{marker}")

    def _set_project_dirty(self, dirty: bool = True) -> None:
        """Track whether the in-memory scientific session differs from the saved project."""

        if self._restoring_project:
            return
        self._project_dirty = bool(dirty)
        self._update_window_title()

    def _recent_project_paths(self) -> list[Path]:
        raw = self._settings.value("recentProjects", [])
        entries = [raw] if isinstance(raw, str) else list(raw or [])
        return [Path(str(entry)) for entry in entries if str(entry).strip()][:8]

    def _remember_recent_project(self, path: Path) -> None:
        resolved = path.resolve()
        paths = [candidate for candidate in self._recent_project_paths() if candidate.resolve() != resolved]
        paths.insert(0, resolved)
        self._settings.setValue("recentProjects", [str(candidate) for candidate in paths[:8]])
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        if not hasattr(self, "_recent_projects_menu"):
            return
        self._recent_projects_menu.clear()
        paths = self._recent_project_paths()
        if not paths:
            empty = self._recent_projects_menu.addAction("No recent projects")
            empty.setEnabled(False)
            return
        for path in paths:
            action = self._recent_projects_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _checked=False, selected=path: self._open_project_path(selected))

    def _discard_autosave(self) -> None:
        try:
            self._recovery_path.unlink(missing_ok=True)
        except OSError:
            return
        if hasattr(self, "recover_project_action"):
            self.recover_project_action.setEnabled(False)



    def _install_table_copy_menu(self, table: QTableWidget) -> None:
        """Give every data table the same clipboard-first context menu."""

        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(lambda position, target=table: self._show_table_context_menu(target, position))

    def _show_table_context_menu(self, table: QTableWidget, position) -> None:  # type: ignore[no-untyped-def]
        menu = QMenu(table)
        selected = bool(table.selectedIndexes())
        copy_selection = menu.addAction("Copy selected cells")
        copy_selection.setEnabled(selected)
        copy_selection.triggered.connect(lambda: self._copy_table_text(table, selected_only=True))
        menu.addAction("Copy full table as TSV").triggered.connect(lambda: self._copy_table_text(table, selected_only=False))
        menu.addSeparator()
        menu.addAction("Copy table as image").triggered.connect(lambda: self._copy_widget_image(table, "Table image"))
        menu.exec(table.viewport().mapToGlobal(position))

    def _install_plot_copy_menu(self, plot: pg.PlotWidget) -> None:
        """Offer a simple clipboard image action for either scientific plot."""

        plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        plot.customContextMenuRequested.connect(lambda position, target=plot: self._show_plot_context_menu(target, position))

    def _show_plot_context_menu(self, plot: pg.PlotWidget, position) -> None:  # type: ignore[no-untyped-def]
        menu = QMenu(plot)
        menu.addAction("Copy plot as image").triggered.connect(lambda: self._copy_widget_image(plot, "Plot image"))
        menu.exec(plot.mapToGlobal(position))

    @staticmethod
    def _table_cell_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        if item is not None:
            return item.text()
        widget = table.cellWidget(row, column)
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return ""

    def _copy_table_text(self, table: QTableWidget, *, selected_only: bool) -> None:
        """Copy a rectangular table selection or the whole table as TSV."""

        indexes = table.selectedIndexes() if selected_only else []
        if selected_only and not indexes:
            return
        if selected_only:
            rows = range(min(index.row() for index in indexes), max(index.row() for index in indexes) + 1)
            columns = range(min(index.column() for index in indexes), max(index.column() for index in indexes) + 1)
            selected_cells = {(index.row(), index.column()) for index in indexes}
        else:
            rows = range(table.rowCount())
            columns = range(table.columnCount())
            selected_cells = None
        lines: list[str] = []
        if not selected_only:
            lines.append("\t".join(table.horizontalHeaderItem(column).text() if table.horizontalHeaderItem(column) else "" for column in columns))
        for row in rows:
            lines.append(
                "\t".join(
                    self._table_cell_text(table, row, column) if selected_cells is None or (row, column) in selected_cells else ""
                    for column in columns
                )
            )
        QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage("Copied selected table cells." if selected_only else "Copied full table as TSV.", 4_000)

    def _copy_widget_image(self, widget: QWidget, description: str) -> None:
        """Copy the current visible table or plot surface as a clipboard image."""

        QApplication.clipboard().setPixmap(widget.grab())
        self.statusBar().showMessage(f"Copied {description.lower()} to the clipboard.", 4_000)

    @staticmethod
    def _add_glow(widget: QWidget, colour: QColor, *, blur_radius: int = 18) -> None:
        """Retained compatibility hook; the modern 2D theme uses no glow."""

        del colour, blur_radius
        widget.setGraphicsEffect(None)

    def _create_plot_card(self, kicker: str, title: str, plot: pg.PlotWidget) -> QFrame:
        card = QFrame()
        card.setObjectName("plotCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(4)
        kicker_label = QLabel(kicker)
        kicker_label.setObjectName("plotKicker")
        title_label = QLabel(title)
        title_label.setObjectName("plotTitle")
        layout.addWidget(kicker_label)
        layout.addWidget(title_label)
        layout.addSpacing(6)
        layout.addWidget(plot, stretch=1)
        return card

    @staticmethod
    def _configure_plot(plot: pg.PlotWidget, axis_label: str, unit: str) -> None:
        plot.setBackground("#141718")
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel("left", axis_label.upper(), units=unit, color="#cdd3d1")
        plot.setLabel("bottom", "BINDING ENERGY", units="eV", color="#cdd3d1")
        plot.getPlotItem().getViewBox().setBorder(pg.mkPen("#3f4749", width=1))
        tick_font = QFont("Segoe UI", 9)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("#7f898b", width=1))
            axis.setTextPen(pg.mkPen("#c7cecc"))
            axis.setTickFont(tick_font)
            axis.setStyle(tickTextOffset=8)

    def open_spectrum(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open XPS spectrum",
            self._last_directory,
            "Supported spectra (*.csv *.txt *.tsv *.dat *.vgd *.VGD);;Thermo/VG VGD (*.vgd *.VGD);;Text spectra (*.csv *.txt *.tsv *.dat);;All files (*)",
        )
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".vgd":
                spectra = read_vgd_spectra(path)
                spectrum = self._select_vgd_spectrum(spectra)
                if spectrum is None:
                    return
            else:
                spectrum = read_delimited_spectrum(path)
        except (OSError, VGDReadError, ValueError) as error:
            self.show_error("Could not open spectrum", str(error))
            return
        self._last_directory = str(Path(path).parent)
        self.add_workspace_spectra([spectrum], activate=True)
        if spectrum.energy_range[1] - spectrum.energy_range[0] <= 150.0:
            self.add_peak()
        else:
            self.statusBar().showMessage(
                "Loaded a wide survey. Run Find possible elements, then seed reviewed candidates for fitting."
            )

    def _select_vgd_spectrum(self, spectra: list[Spectrum]) -> Spectrum | None:
        if len(spectra) == 1:
            return spectra[0]
        labels = [f"Spectrum {index + 1} — {spectrum.size} points" for index, spectrum in enumerate(spectra)]
        selected, accepted = QInputDialog.getItem(self, "Select VGD spectrum", "Spectrum:", labels, 0, False)
        if not accepted:
            return None
        return spectra[labels.index(selected)]

    def load_synthetic_example(self) -> None:
        spectrum, _ = synthetic_doublet()
        self.add_workspace_spectra([spectrum], activate=True)
        components, ties = make_spin_orbit_doublet(
            "Synthetic 3d",
            center=159.7,
            fwhm=1.55,
            area=760.0,
            splitting=1.75,
            upper_to_lower_area_ratio=0.5,
            profile=PeakProfile.PSEUDO_VOIGT,
            eta=0.25,
            center_bounds=(158.5, 161.0),
            fwhm_bounds=(0.5, 3.0),
            area_bounds=(1.0, 2_000.0),
        )
        self._ties = ties
        for component in components:
            self.add_peak_row(component)
        self._store_active_workspace_fit_state()
        self.statusBar().showMessage("Loaded a synthetic doublet with physical parameter ties.")

    def set_spectrum(self, spectrum: Spectrum, *, clear_components: bool) -> None:
        self.spectrum = spectrum
        self._recommend_fit_weighting_for_spectrum(spectrum)
        self.fit_result = None
        self._spectrum_undo.clear()
        self._reset_survey_results(reset_preferences=True)
        if clear_components:
            self.peak_table.setRowCount(0)
            self._ties = []
        self._update_window_title()
        self.recalculate_background()



    def render_spectrum(self) -> None:
        self.spectrum_plot.clear()
        self.residual_plot.clear()
        if self.spectrum is None:
            return
        x, y = self.spectrum.energy, self.spectrum.intensity
        axis_title = "BINDING ENERGY" if self.spectrum.energy_scale is EnergyScale.BINDING else "KINETIC ENERGY"
        self.spectrum_plot.setLabel("bottom", axis_title, units="eV", color="#d9dfdd")
        self.residual_plot.setLabel("bottom", axis_title, units="eV", color="#d9dfdd")
        legend = self.spectrum_plot.addLegend(offset=(12, 12))
        legend.setBrush(pg.mkBrush(24, 29, 31, 230))
        legend.setPen(pg.mkPen("#647279", width=1))
        self.spectrum_plot.plot(x, y, pen=pg.mkPen(221, 231, 229, 36, width=5), antialias=True)
        measured = self.spectrum_plot.plot(x, y, pen=pg.mkPen("#edf2ef", width=1.35), name="Measured")
        measured.setZValue(7)
        if self.spectrum.correction_history:
            original_axis = original_axis_in_active_scale(self.spectrum)
            if original_axis is not None:
                self.spectrum_plot.plot(
                    original_axis,
                    y,
                    pen=pg.mkPen("#8995ad", width=1.0, style=Qt.PenStyle.DashLine),
                    name="Uncorrected axis",
                )
        if self.background is not None:
            self.spectrum_plot.plot(x, self.background, pen=pg.mkPen(216, 171, 98, 42, width=5), antialias=True)
            background_curve = self.spectrum_plot.plot(x, self.background, pen=pg.mkPen("#d7aa62", width=1.4), name="Background")
            background_curve.setZValue(3)
        if self.survey_analysis is not None:
            marked_peaks: set[float] = set()
            for match in self.survey_analysis.matches:
                peak_energy = match.peak.binding_energy
                if peak_energy in marked_peaks:
                    continue
                marked_peaks.add(peak_energy)
                self.spectrum_plot.addItem(
                    pg.InfiniteLine(
                        pos=peak_energy,
                        angle=90,
                        pen=pg.mkPen("#d7b574", width=1.0, style=Qt.PenStyle.DotLine),
                    )
                )
        if self.fit_result is not None:
            colours = ("#ffb05d", "#ee7b6d", "#d896f4", "#6dd6c0", "#ffd166", "#8aa9ff", "#e58cbc")
            for index, (name, curve) in enumerate(self.fit_result.components.items()):
                colour = QColor(colours[index % len(colours)])
                component = curve + self.fit_result.background
                lower = self.spectrum_plot.plot(x, self.fit_result.background, pen=pg.mkPen(0, 0, 0, 0))
                upper = self.spectrum_plot.plot(x, component, pen=pg.mkPen(colour, width=1.45), name=name)
                component_fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(colour.red(), colour.green(), colour.blue(), 74))
                component_fill.setZValue(1)
                self.spectrum_plot.addItem(component_fill)
                lower.setZValue(2)
                upper.setZValue(4)
            self.spectrum_plot.plot(x, self.fit_result.fitted_intensity, pen=pg.mkPen(236, 242, 237, 46, width=7), antialias=True)
            fit_envelope = self.spectrum_plot.plot(
                x, self.fit_result.fitted_intensity, pen=pg.mkPen("#f5f0df", width=2.05), name="Fit envelope"
            )
            fit_envelope.setZValue(6)
            self.residual_plot.addLine(y=0.0, pen=pg.mkPen("#847465", width=1.0))
            self.residual_plot.plot(x, self.fit_result.residual, pen=pg.mkPen(245, 118, 109, 44, width=5), antialias=True)
            self.residual_plot.plot(x, self.fit_result.residual, pen=pg.mkPen("#ef8a75", width=1.25))
        self.spectrum_plot.autoRange()
        self.residual_plot.autoRange()

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "About XPS-master",
            f"Created and maintained by {AUTHOR_NAME}\n{AUTHOR_EMAIL}\n{AUTHOR_WEBSITE}\n\n"
            "Phase 14 strengthens the scientific backbone: compiled analytic peak derivatives, scaled optimization, raw-count Poisson maximum likelihood, likelihood-consistent AICc/BIC, propagated tie bounds, and SVD/Fisher uncertainty estimates. Rank, correlation, and boundary warnings are preserved in projects and Excel exports. Global center uncertainties include their covariance with relative energy shifts.\n\n"
            "Phase 13 adds Global Multi-Spectrum Fitting: selected spectra from one high-resolution series are fitted simultaneously with shared physical peak parameters, spectrum-specific areas, and bounded relative energy shifts. Independent reference fits are retained for AICc/BIC comparison, and the dashboard exposes every fit, residual, shift, uncertainty, and component trend before the analyst chooses whether to apply the global curves.\n\n"
            "Phase 12 adds Physics-Constrained Model Comparison: explicit background, line-shape, and neighbouring peak-count hypotheses are fitted in a cancellable worker and ranked with AICc/BIC support, residual structure, boundary, overlap, background, and parameter-correlation diagnostics. Every alternative is retained until the analyst explicitly accepts one; numerical recommendation is never presented as chemical-state proof. Generic p/d/f degeneracy templates create auditable spin-orbit constraints while leaving the experimental splitting under analyst control.\n\n"
            "Phase 11 adds Stability-Consensus: a cancellable perturbation ensemble stress-tests requested components across noise, calibration uncertainty, ROI boundaries, background variants, starting positions, line shapes, and neighbouring peak counts. Component survival, sensitivity, correlation, model support, and Trust Scores are reported without replacing the retained fit or asserting chemistry.\n\n"
            "Phase 10D adds project-wide energy calibration: one signed eV shift can be applied atomically to every loaded spectrum while imported axes, uncertainty, and provenance remain retained. Saved fit centers and constraints move with the calibrated axes, fit/residual intensities remain intact, and prior survey screens are recalculated.\n\n"
            "Phase 10C adds complete Excel export: one Summary plus one worksheet per workspace plot, with native fit/residual charts, measured and original axes, background, formula-driven net signal and residual, every deconvoluted component, fitted parameters and uncertainties, constraints, histories, survey evidence, and acquisition metadata.\n\n"
            "Phase 10B adds complete workspace persistence: every loaded spectrum, editable fit model, retained fit curve, survey result, background configuration, and active panel is saved in one portable project. Save/Save As, recent projects, autosave recovery, and unsaved-change protection keep long analyses safe.\n\n"
            "Phase 9 adds residual-aware smart deconvolution: the analyst selects a component count and profile, then a background worker evaluates diverse bounded starting models with live progress. It ranks numerical candidates using BIC, residual structure, and collapsed-peak penalties.\n\n"
            "Phase 8 retained per-card fits, a tabbed analysis workflow, and a metallic research-console interface alongside reproducible backgrounds, processing, survey/quantification tools, and configurable peak line shapes.\n\n"
            "Mathematical curves and the Doniach-Sunjic metallic-state model are deliberately labelled separately. Instrument-specific factors, backgrounds, fits, and assignments must be reviewed by a qualified analyst.",
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Avoid destroying a running fit worker while its bounded candidate is evaluating."""

        if self._smart_thread is not None and self._smart_thread.isRunning():
            self.cancel_smart_deconvolution()
            self.statusBar().showMessage("Stopping smart deconvolution before closing. Close again when the worker has stopped.")
            event.ignore()
            return
        if self._consensus_thread is not None and self._consensus_thread.isRunning():
            self.cancel_stability_consensus()
            self.statusBar().showMessage("Stopping stability consensus before closing. Close again when it has stopped.")
            event.ignore()
            return
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            self.cancel_model_comparison()
            self.statusBar().showMessage("Stopping model comparison before closing. Close again when it has stopped.")
            event.ignore()
            return
        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            self.cancel_global_series_fit()
            self.statusBar().showMessage("Stopping global fitting before closing. Close again when it has stopped.")
            event.ignore()
            return
        if self._project_dirty and self.spectrum is not None:
            if not self._confirm_session_replacement():
                event.ignore()
                return
            if self._project_dirty:
                self._discard_autosave()
        event.accept()

    def show_error(self, title: str, message: str) -> None:
        self.statusBar().showMessage(message, 8_000)
        QMessageBox.warning(self, title, message)
