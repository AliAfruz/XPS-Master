import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from xps_afruz.ui import main_window as main_window_module
from xps_afruz.ui import project_controller as project_controller_module
from xps_afruz.ui.main_window import MainWindow
from xps_afruz.fitting.model_comparison import ModelComparisonSettings, build_standard_candidates, compare_models
from xps_afruz.fitting.stability_consensus import ComponentStability, StabilityConsensusResult


class _MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.mark.gui
def test_main_window_has_responsive_tabbed_analysis_docks(application: QApplication) -> None:
    window = MainWindow()
    window.resize(1_366, 768)
    window.show()
    application.processEvents()

    try:
        assert [window.workflow_tabs.tabText(index) for index in range(window.workflow_tabs.count())] == [
            "Fit",
            "Survey",
            "Workspace",
            "Publish",
        ]
        assert window.quant_dock in window.tabifiedDockWidgets(window.survey_dock)
        assert [action.text() for action in window.view_menu.actions()] == [
            "Survey material finder",
            "Atomic composition",
            "Multi-spectrum workspace",
        ]
        initial_width = window.centralWidget().width()
        window.survey_dock.hide()
        window.quant_dock.hide()
        application.processEvents()
        assert window.centralWidget().width() > initial_width
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_workspace_board_opens_only_for_multi_spectrum_context(application: QApplication) -> None:
    window = MainWindow()
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        assert not window.global_start_button.isEnabled()
        assert window.workspace_dock.isHidden()
        window.load_synthetic_example()
        application.processEvents()
        assert window.workspace_dock.isVisible()
        assert len(window._workspace_spectra) == 2
        assert window.global_start_button.isEnabled()
        assert any(
            action.text() == "Run global multi-spectrum fit"
            for action in window.findChildren(main_window_module.QAction)
        )
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_complete_workspace_can_be_saved_and_reopened(application: QApplication, tmp_path) -> None:
    source = MainWindow()
    restored = MainWindow()
    source._settings = _MemorySettings()
    restored._settings = _MemorySettings()
    source.show()
    restored.show()
    application.processEvents()
    destination = tmp_path / "retained-workspace.xpsa"

    try:
        source.load_synthetic_example()
        source.load_synthetic_example()
        source.workflow_tabs.setCurrentIndex(2)
        assert source._write_project(destination)

        assert restored._open_project_path(destination)
        assert len(restored._workspace_spectra) == 2
        assert restored._active_workspace_index == 1
        assert len(restored._workspace_peak_rows[0]) == 2
        assert len(restored._workspace_peak_rows[1]) == 2
        assert restored.workflow_tabs.currentIndex() == 2
        assert not restored._project_dirty
        restored.activate_workspace_spectrum(0)
        assert restored.peak_table.rowCount() == 2
    finally:
        source._set_project_dirty(False)
        restored._set_project_dirty(False)
        source.close()
        restored.close()


@pytest.mark.gui
def test_dirty_workspace_creates_recoverable_autosave(application: QApplication, tmp_path) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    window._recovery_path = tmp_path / "workspace-autosave.xpsa"
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        assert window._project_dirty
        window._autosave_project()
        assert window._recovery_path.exists()

        recovered = MainWindow()
        recovered._settings = _MemorySettings()
        recovered._recovery_path = window._recovery_path
        try:
            assert recovered._open_project_path(window._recovery_path, recovered=True)
            assert len(recovered._workspace_spectra) == 1
            assert recovered._current_project_path is None
            assert recovered._project_dirty
        finally:
            recovered._set_project_dirty(False)
            recovered.close()
    finally:
        window._discard_autosave()
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_excel_export_action_writes_the_complete_workspace(application: QApplication, tmp_path, monkeypatch) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    destination = tmp_path / "gui-workspace-export.xlsx"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "Excel workbook (*.xlsx)"),
    )
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        window.load_synthetic_example()
        assert any(action.text() == "Export complete workspace to Excel…" for action in window.findChildren(main_window_module.QAction))
        assert window.export_workspace_excel()
        assert destination.exists()
        assert destination.stat().st_size > 10_000
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_manuscript_plotter_launcher_uses_the_active_python_environment(
    application: QApplication,
    tmp_path,
    monkeypatch,
) -> None:
    window = MainWindow()
    calls: list[tuple[str, list[str]]] = []
    workbook = tmp_path / "complete-analysis.xlsx"
    workbook.write_bytes(b"PK test placeholder")
    monkeypatch.setattr(
        project_controller_module.QProcess,
        "startDetached",
        lambda program, arguments: calls.append((program, list(arguments))) or True,
    )

    try:
        actions = [action.text() for action in window.findChildren(main_window_module.QAction)]
        assert "Export workspace and open…" in actions
        assert "Open Avantage or XPS-master Excel…" in actions
        assert window._launch_manuscript_plotter(workbook)
        assert calls == [
            (
                project_controller_module.sys.executable,
                [
                    "-m",
                    "xps_afruz.manuscript_plotter.studio",
                    str(workbook.resolve()),
                ],
            )
        ]
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_publish_runner_panel_tracks_workspace_and_launches_empty_plotter(
    application: QApplication,
    monkeypatch,
) -> None:
    window = MainWindow()
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        project_controller_module.QProcess,
        "startDetached",
        lambda program, arguments: calls.append((program, list(arguments))) or True,
    )

    try:
        window.workflow_tabs.setCurrentIndex(3)
        assert window.workflow_tabs.tabText(3) == "Publish"
        assert not window.manuscript_export_button.isEnabled()
        assert "No project spectra" in window.manuscript_runner_status.text()

        assert window.launch_manuscript_plotter()
        assert calls == [
            (
                project_controller_module.sys.executable,
                ["-m", "xps_afruz.manuscript_plotter.studio"],
            )
        ]
        assert "empty XPS-master Manuscript Plotter" in window.manuscript_runner_status.text()

        window.load_synthetic_example()
        assert window.manuscript_export_button.isEnabled()
        assert "Ready: 1 spectrum" in window.manuscript_runner_status.text()
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_project_energy_shift_updates_every_card_and_retains_fit(application: QApplication) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        window.run_fit()
        first_fit = window._workspace_fit_results[0]
        assert first_fit is not None
        first_curve = first_fit.fitted_intensity.copy()
        window.load_synthetic_example()
        original_axes = [spectrum.original_energy.copy() for spectrum in window._workspace_spectra]
        active_axes = [spectrum.energy.copy() for spectrum in window._workspace_spectra]
        centers = [[float(row["center"]) for row in rows] for rows in window._workspace_peak_rows]
        seed_centers = [
            [float(row.get("seed_center", row["center"])) for row in rows]
            for rows in window._workspace_peak_rows
        ]

        assert window.apply_project_energy_shift(
            1.125,
            reference_label="Shared Au 4f reference",
            uncertainty_eV=0.04,
        ) == 2

        for index, spectrum in enumerate(window._workspace_spectra):
            assert np.allclose(spectrum.energy, active_axes[index] + 1.125)
            assert np.array_equal(spectrum.original_energy, original_axes[index])
            assert spectrum.correction_history[-1]["method"] == "project_wide_manual_shift"
            assert spectrum.correction_history[-1]["affected_spectra"] == 2
            assert [float(row["center"]) for row in window._workspace_peak_rows[index]] == pytest.approx(
                [center + 1.125 for center in centers[index]]
            )
            assert [float(row["seed_center"]) for row in window._workspace_peak_rows[index]] == pytest.approx(
                [center + 1.125 for center in seed_centers[index]]
            )
        retained = window._workspace_fit_results[0]
        assert retained is not None
        assert np.array_equal(retained.fitted_intensity, first_curve)
        assert retained.parameter_values["Synthetic 3d lower.center"] == pytest.approx(
            first_fit.parameter_values["Synthetic 3d lower.center"] + 1.125
        )
        assert window.spectrum is window._workspace_spectra[1]
        assert window._project_dirty
        assert any(
            action.text() == "Apply signed shift to entire project…"
            for action in window.findChildren(main_window_module.QAction)
        )
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_manual_seed_anchors_survive_fit_output_and_follow_explicit_center_edits(
    application: QApplication,
) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        assert window.seed_guardrails_checkbox.isChecked()
        original_seeds = [float(row["seed_center"]) for row in window.peak_rows()]

        window.run_fit()

        fitted_rows = window.peak_rows()
        assert [float(row["seed_center"]) for row in fitted_rows] == pytest.approx(original_seeds)
        assert "Smart seed protection: ON" in window.fit_summary.text()

        center_item = window.peak_table.item(0, 1)
        assert center_item is not None
        edited_center = float(center_item.text()) + 0.15
        center_item.setText(f"{edited_center:.6g}")
        application.processEvents()

        assert float(window.peak_rows()[0]["seed_center"]) == pytest.approx(edited_center)
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_consensus_result_is_retained_on_its_workspace_card_without_replacing_fit(
    application: QApplication, monkeypatch
) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        window.run_fit()
        retained_fit = window.fit_result
        assert retained_fit is not None
        component = ComponentStability(
            "Consensus Peak 1", 12, 0.95, 0.90, 160.1, 0.04, 1.25, 0.06, 900.0, 20.0, 0.022,
            0.10, "LOW", 0.20, "MODERATE", 0.02, 0.0, 0.0, 0.40, 89.0, "STRONG NUMERICAL SUPPORT"
        )
        result = StabilityConsensusResult(
            spectrum_label=window.spectrum.label,
            requested_peak_count=2,
            peak_count_probabilities={1: 0.02, 2: 0.93, 3: 0.05},
            bic_peak_count_probabilities={1: 0.01, 2: 0.95, 3: 0.04},
            aicc_peak_count_probabilities={1: 0.03, 2: 0.91, 3: 0.06},
            bic_by_peak_count={1: 100.0, 2: 80.0, 3: 86.0},
            aicc_by_peak_count={1: 98.0, 2: 78.0, 3: 84.0},
            components=(component,),
            requested_runs=6,
            successful_runs=17,
            failed_runs=1,
            paired_model_scenarios=5,
            candidate_trials_per_run=4,
            random_seed=73109,
            background_models=("Linear", "Shirley"),
            profiles=("gaussian", "pseudo_voigt"),
            weighting="uniform",
            count_scale=None,
            calibration_uncertainty_eV=0.04,
            roi_jitter_fraction=0.025,
            bootstrap_strength=1.0,
            calibration_sigma_scale=1.0,
            max_nfev=900,
            max_background_above_fraction=0.12,
            numerical_note="Trust scores do not prove chemistry.",
        )
        window._consensus_target_spectrum = window.spectrum
        window._consensus_workspace_index = window._active_workspace_index
        window._apply_consensus_result(result)

        assert window.fit_result is retained_fit
        assert window._workspace_consensus_results[0] is result
        assert window.consensus_report_button.isEnabled()
        assert "Trust Report ready" in window.consensus_status.text()
        assert "Trust:" in window.workspace_list.item(0).text()
        assert any(
            action.text() == "Open Stability Trust Report"
            for action in window.findChildren(main_window_module.QAction)
        )
        monkeypatch.setattr(main_window_module.QDialog, "exec", lambda _dialog: 0)
        window.show_stability_consensus_report()
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_model_comparison_retains_alternatives_until_explicit_acceptance(
    application: QApplication,
    monkeypatch,
) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        window.run_fit()
        retained_fit = window.fit_result
        assert retained_fit is not None
        settings = ModelComparisonSettings(
            compare_backgrounds=False,
            compare_profiles=True,
            compare_neighbor_counts=False,
            max_nfev=500,
            weighting=window.fit_weighting,
            count_scale=window.poisson_count_scale,
        )
        candidates = build_standard_candidates(
            window.spectrum,
            tuple(window.components_from_table()),
            tuple(window.active_ties()),
            window.background_settings,
            settings,
        )
        result = compare_models(window.spectrum, candidates, settings)
        window._comparison_target_spectrum = window.spectrum
        window._comparison_workspace_index = window._active_workspace_index
        window._apply_model_comparison_result(result)

        assert window.fit_result is retained_fit
        assert window._workspace_model_comparisons[0] is result
        assert window.comparison_report_button.isEnabled()
        assert "Models:" in window.workspace_list.item(0).text()
        monkeypatch.setattr(main_window_module.QDialog, "exec", lambda _dialog: 0)
        window.show_model_comparison_report()

        selected = result.recommended
        window.accept_model_comparison_candidate(selected.candidate.identifier)
        assert window.fit_result is selected.fit_result
        assert window._workspace_model_comparisons[0] is result
        assert window.peak_table.rowCount() == selected.component_count
        assert "Accepted comparison candidate" in window.fit_summary.text()
    finally:
        window._set_project_dirty(False)
        window.close()


@pytest.mark.gui
def test_model_comparison_worker_reports_progress_and_retains_complete_batch(application: QApplication) -> None:
    window = MainWindow()
    window._settings = _MemorySettings()
    window.show()
    application.processEvents()

    try:
        window.load_synthetic_example()
        window.comparison_backgrounds.setChecked(False)
        window.comparison_profiles.setChecked(True)
        window.comparison_neighbors.setChecked(False)
        window.comparison_max_nfev.setValue(500)
        window.start_model_comparison()
        deadline = time.monotonic() + 15.0
        while window._comparison_thread is not None and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.01)
        application.processEvents()

        assert window._comparison_thread is None
        assert window._model_comparison_result is not None
        assert len(window._model_comparison_result.candidates) == 2
        assert window.comparison_progress.value() == 100
        assert window.comparison_report_button.isEnabled()
        assert window.fit_result is None
    finally:
        if window._comparison_thread is not None and window._comparison_thread.isRunning():
            window.cancel_model_comparison()
            window._comparison_thread.quit()
            window._comparison_thread.wait(5_000)
        window._set_project_dirty(False)
        window.close()
