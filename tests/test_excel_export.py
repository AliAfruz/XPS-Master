from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from xps_afruz import Spectrum
from xps_afruz.backgrounds import BackgroundDiagnostics, BackgroundMethod, BackgroundSettings
from xps_afruz.fitting.model import FitResult, PeakComponent
from xps_afruz.fitting.global_series import (
    GlobalComponentTrend,
    GlobalSeriesResult,
    GlobalSeriesSettings,
    GlobalSpectrumFit,
)
from xps_afruz.fitting.model_comparison import ModelCandidate, ModelCandidateEvaluation, ModelComparisonResult
from xps_afruz.fitting.peak_shapes import PeakProfile
from xps_afruz.fitting.stability_consensus import ComponentStability, StabilityConsensusResult
from xps_afruz.io.excel_export import export_workspace_to_excel
from xps_afruz.io.project_io import WorkspaceItemData
from xps_afruz.manuscript_plotter.core import (
    PlotSettings,
    default_trace_styles,
    list_xlsx_sheets,
    load_xps_sheet,
    make_figure,
    transformed_series,
)


def _retained_fit() -> tuple[Spectrum, FitResult]:
    energy = np.linspace(700.0, 705.0, 6)
    background = np.linspace(2.0, 2.5, 6)
    first = np.array([0.2, 1.8, 4.0, 1.7, 0.2, 0.0])
    second = np.array([0.0, 0.1, 0.6, 2.2, 1.1, 0.1])
    fitted = background + first + second
    residual = np.array([0.05, -0.10, 0.08, -0.03, 0.04, -0.02])
    measured = fitted - residual
    spectrum = Spectrum(
        energy,
        measured,
        label="Fe 2p / fitted scan",
        intensity_unit="kcounts",
        photon_energy_eV=1486.68,
        energy_uncertainty_eV=0.08,
        metadata={"pass_energy_eV": 20.0, "source": "Mono Al K-alpha"},
        processing_history=({"operation": "savitzky_golay", "window": 5, "polynomial": 2},),
    )
    result = FitResult(
        success=True,
        message="converged",
        parameter_values={
            "Fe metal.center": 702.0,
            "Fe metal.fwhm": 1.2,
            "Fe metal.area": 7.5,
            "Fe metal.eta": 0.35,
            "Fe metal.asymmetry": 0.0,
            "Fe metal.instrument_fwhm": 0.0,
            "Fe oxide.center": 703.2,
            "Fe oxide.fwhm": 1.5,
            "Fe oxide.area": 5.0,
            "Fe oxide.eta": 0.45,
            "Fe oxide.asymmetry": 0.0,
            "Fe oxide.instrument_fwhm": 0.0,
        },
        standard_errors={
            "Fe metal.center": 0.03,
            "Fe metal.fwhm": 0.05,
            "Fe metal.area": 0.20,
            "Fe oxide.center": 0.04,
            "Fe oxide.fwhm": 0.07,
            "Fe oxide.area": 0.25,
        },
        background=background,
        components={"Fe metal": first, "Fe oxide": second},
        fitted_intensity=fitted,
        residual=residual,
        weighted_residual=residual / np.sqrt(measured),
        chi_square=1.2,
        reduced_chi_square=0.6,
        degrees_of_freedom=2,
        optimizer_evaluations=146,
        weighting="poisson",
        count_scale=1000.0,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        statistical_chi_square=1.2,
        reduced_statistical_chi_square=0.6,
        uncertainty_model="Pearson Poisson",
    )
    return spectrum, result


def _peak(name: str, center: float, fwhm: float, area: float, eta: float) -> dict[str, object]:
    return {
        "name": name,
        "center": center,
        "fwhm": fwhm,
        "area": area,
        "profile": "pseudo_voigt",
        "eta": eta,
        "asymmetry": 0.0,
        "instrument_fwhm": 0.0,
    }


def _trust_report(label: str) -> StabilityConsensusResult:
    component = ComponentStability(
        "Consensus Peak 1", 12, 0.92, 0.88, 702.0, 0.05, 1.2, 0.08, 7.5, 0.4, 0.053,
        0.12, "LOW", 0.25, "MODERATE", 0.03, 0.0, 0.0, 0.42, 87.5, "STRONG NUMERICAL SUPPORT"
    )
    return StabilityConsensusResult(
        spectrum_label=label,
        requested_peak_count=1,
        peak_count_probabilities={1: 0.88, 2: 0.12},
        bic_peak_count_probabilities={1: 0.90, 2: 0.10},
        aicc_peak_count_probabilities={1: 0.86, 2: 0.14},
        bic_by_peak_count={1: 10.0, 2: 14.0},
        aicc_by_peak_count={1: 9.0, 2: 12.0},
        components=(component,),
        requested_runs=6,
        successful_runs=11,
        failed_runs=1,
        paired_model_scenarios=5,
        candidate_trials_per_run=4,
        random_seed=73109,
        background_models=("Linear", "Shirley"),
        profiles=("gaussian",),
        weighting="uniform",
        count_scale=None,
        calibration_uncertainty_eV=0.08,
        roi_jitter_fraction=0.025,
        bootstrap_strength=1.0,
        calibration_sigma_scale=1.0,
        max_nfev=900,
        max_background_above_fraction=0.12,
        numerical_note="Trust scores do not prove chemistry.",
    )


def _model_comparison(label: str, fit: FitResult) -> ModelComparisonResult:
    components = (
        PeakComponent("Fe metal", 702.0, 1.2, 7.5, PeakProfile.PSEUDO_VOIGT, eta=0.35),
        PeakComponent("Fe oxide", 703.2, 1.5, 5.0, PeakProfile.PSEUDO_VOIGT, eta=0.45),
    )
    backgrounds = BackgroundSettings(BackgroundMethod.LINEAR, {"edge_fraction": 0.03})
    diagnostics = BackgroundDiagnostics(2.0, 2.5, 0.0, 0.0, 0.05)
    evaluations = []
    for rank, (identifier, candidate_label, support) in enumerate(
        (("analyst", "Analyst model", 0.82), ("gaussian", "Gaussian alternative", 0.18)),
        start=1,
    ):
        candidate_components = components if identifier == "analyst" else tuple(
            PeakComponent(component.name, component.center, component.fwhm, component.area, PeakProfile.GAUSSIAN)
            for component in components
        )
        candidate = ModelCandidate(
            identifier,
            candidate_label,
            candidate_components,
            (),
            backgrounds,
            "Retained test hypothesis.",
        )
        evaluations.append(
            ModelCandidateEvaluation(
                candidate,
                fit,
                diagnostics,
                6,
                20.0 + rank,
                22.0 + rank,
                support,
                support,
                support,
                0.05,
                1.9,
                0.0,
                0.0,
                0.4,
                "STRONG NUMERICAL SUPPORT",
                True,
                rank,
                (),
            )
        )
    return ModelComparisonResult(
        label,
        tuple(evaluations),
        "analyst",
        "poisson",
        1000.0,
        "Model support compares numerical candidates and does not prove chemistry.",
    )


def test_excel_export_contains_one_data_sheet_per_plot_with_charts_and_formulas(tmp_path: Path) -> None:
    spectrum, fit = _retained_fit()
    survey = Spectrum(
        np.linspace(0.0, 5.0, 6),
        [2.0, 3.0, 2.5, 5.0, 2.7, 2.2],
        label="Survey: overview?",
        metadata={"dwell_time_s": 0.1},
    )
    items = (
        WorkspaceItemData(
            spectrum=spectrum,
            background_method=BackgroundMethod.LINEAR,
            background_parameters={"edge_fraction": 0.03},
            peaks=(
                _peak("Fe metal", 702.0, 1.2, 7.5, 0.35),
                _peak("Fe oxide", 703.2, 1.5, 5.0, 0.45),
            ),
            ties=({"target": "Fe oxide.fwhm", "source": "Fe metal.fwhm", "scale": 1.0, "offset": 0.3},),
            fit_result=fit,
            consensus_result=_trust_report(spectrum.label),
            model_comparison=_model_comparison(spectrum.label, fit),
        ),
        WorkspaceItemData(
            spectrum=survey,
            background_method=BackgroundMethod.LINEAR,
            background_parameters={"edge_fraction": 0.03},
        ),
    )
    destination = tmp_path / "complete-analysis.xlsx"

    sheet_names = export_workspace_to_excel(
        destination,
        items,
        active_workspace_index=0,
        project_name="Excel export test",
    )

    assert destination.read_bytes()[:2] == b"PK"
    assert sheet_names == ("01 Fe 2p fitted scan", "02 Survey overview")
    assert not (tmp_path / "complete-analysis.tmp.xlsx").exists()
    with ZipFile(destination) as archive:
        names = set(archive.namelist())
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        first_sheet_xml = archive.read("xl/worksheets/sheet2.xml").decode("utf-8")
        shared_strings = archive.read("xl/sharedStrings.xml").decode("utf-8")
        assert "Export Summary" in workbook_xml
        assert all(sheet_name in workbook_xml for sheet_name in sheet_names)
        assert "<f>C32-D32</f>" in first_sheet_xml
        assert "<f>D32+SUM(I32,K32)</f>" in first_sheet_xml
        assert "<f>F32-C32</f>" in first_sheet_xml
        assert "Weighted residual" in shared_strings
        assert "Fitted component parameters" in shared_strings
        assert "Processing history" in shared_strings
        assert "Stability-Consensus Trust Report" in shared_strings
        assert "Physics-Constrained Model Comparison" in shared_strings
        assert "Gaussian alternative" in shared_strings
        assert "COMPARE · Analyst model · fit envelope" in shared_strings
        assert "STRONG NUMERICAL SUPPORT" in shared_strings
        assert len([name for name in names if name.startswith("xl/charts/chart")]) == 3

    assert list_xlsx_sheets(destination) == list(sheet_names)
    plotted = load_xps_sheet(destination, sheet_names[0])
    assert plotted.source_format == "XPS-master"
    assert np.array_equal(plotted.energy, spectrum.energy)
    assert np.allclose(plotted.measured, spectrum.intensity)
    assert np.allclose(plotted.envelope, fit.fitted_intensity)
    assert np.allclose(plotted.residual, fit.residual)
    assert list(plotted.components) == ["Fe metal", "Fe oxide"]
    assert plotted.components_include_background
    transformed = transformed_series(plotted, PlotSettings())
    assert np.allclose(transformed["component:Fe metal"], fit.components["Fe metal"])
    assert np.allclose(transformed["component:Fe oxide"], fit.components["Fe oxide"])
    figure, _axis, artists = make_figure(
        plotted,
        PlotSettings(),
        default_trace_styles(plotted),
        [],
    )
    assert len(figure.axes) == 2
    assert artists["residual_axis"] is figure.axes[1]


def test_excel_export_rejects_misaligned_retained_fit(tmp_path: Path) -> None:
    spectrum, fit = _retained_fit()
    object.__setattr__(fit, "residual", np.array([1.0, 2.0]))
    item = WorkspaceItemData(spectrum, BackgroundMethod.LINEAR, {}, fit_result=fit)

    with pytest.raises(ValueError, match="misaligned"):
        export_workspace_to_excel(tmp_path / "bad.xlsx", (item,))

    assert not (tmp_path / "bad.xlsx").exists()


def test_excel_export_includes_global_series_evidence_and_area_chart(tmp_path: Path) -> None:
    spectrum, fit = _retained_fit()
    second = Spectrum(
        spectrum.energy,
        spectrum.intensity,
        label="Fe 2p replicate",
        intensity_unit=spectrum.intensity_unit,
    )
    items = (
        WorkspaceItemData(
            spectrum,
            BackgroundMethod.LINEAR,
            {"edge_fraction": 0.03},
            fit_result=fit,
        ),
        WorkspaceItemData(
            second,
            BackgroundMethod.LINEAR,
            {"edge_fraction": 0.03},
            fit_result=fit,
        ),
    )
    spectrum_fits = (
        GlobalSpectrumFit(0, spectrum.label, 0.0, 0.0, fit, fit),
        GlobalSpectrumFit(1, second.label, 0.08, 0.01, fit, fit),
    )
    trend = GlobalComponentTrend(
        name="Fe metal",
        profile=PeakProfile.PSEUDO_VOIGT.value,
        reference_center_eV=702.0,
        reference_center_stderr_eV=0.03,
        centers_eV=(702.0, 702.08),
        center_stderr_eV=(0.03, 0.032),
        fwhm_eV=(1.2, 1.2),
        fwhm_stderr_eV=(0.05, 0.05),
        areas=(7.5, 8.2),
        area_stderr=(0.2, 0.23),
        eta=(0.35, 0.35),
        asymmetry=(0.0, 0.0),
        instrument_fwhm_eV=(0.0, 0.0),
    )
    global_result = GlobalSeriesResult(
        settings=GlobalSeriesSettings(max_nfev=1_000),
        spectrum_fits=spectrum_fits,
        component_trends=(trend,),
        ties=(),
        success=True,
        message="converged",
        optimizer_evaluations=100,
        total_samples=spectrum.size + second.size,
        global_parameter_count=7,
        independent_parameter_count=10,
        degrees_of_freedom=spectrum.size + second.size - 7,
        global_objective=2.0,
        reduced_global_objective=0.4,
        global_aicc=20.0,
        global_bic=22.0,
        independent_objective=1.8,
        independent_aicc=28.0,
        independent_bic=31.0,
        delta_aicc=8.0,
        delta_bic=9.0,
        preference="Moderate numerical support for the shared-parameter global model",
        numerical_note="Numerical comparison only; analyst review is required.",
    )
    destination = tmp_path / "global-analysis.xlsx"
    export_workspace_to_excel(destination, items, global_series_result=global_result)

    assert list_xlsx_sheets(destination) == ["01 Fe 2p fitted scan", "02 Fe 2p replicate"]

    with ZipFile(destination) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        shared_strings = archive.read("xl/sharedStrings.xml").decode("utf-8")
        chart_xml = "\n".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.startswith("xl/charts/chart")
        )
        assert "Global Series" in workbook_xml
        assert "Delta BIC (independent - global)" in shared_strings
        assert "Component area across the selected series" in chart_xml
        assert len([name for name in archive.namelist() if name.startswith("xl/charts/chart")]) >= 3
