import json

import numpy as np
import pytest

from xps_afruz import Spectrum
from xps_afruz.backgrounds import BackgroundMethod, BackgroundSettings
from xps_afruz.chemistry.survey import (
    DetectedPeak,
    ElementCandidate,
    ExpectedSecondaryLine,
    SurveyAnalysis,
    SurveyMatch,
)
from xps_afruz.chemistry.survey_library import ReferenceLine
from xps_afruz.core.energy import EnergyScale
from xps_afruz.core.energy import charge_correct
from xps_afruz.core.processing import gaussian_smooth
from xps_afruz.fitting.model import FitResult, PeakComponent
from xps_afruz.fitting.model_comparison import ModelCandidate, ModelComparisonSettings, compare_models
from xps_afruz.fitting.peak_shapes import PeakProfile, evaluate_peak
from xps_afruz.fitting.stability_consensus import ComponentStability, StabilityConsensusResult
from xps_afruz.io.project_io import PROJECT_FORMAT_VERSION, WorkspaceItemData, load_project, save_project


def test_project_round_trip_preserves_spectrum_and_fit_setup(tmp_path) -> None:
    spectrum = charge_correct(
        gaussian_smooth(
            Spectrum([3.0, 2.0, 1.0, 0.0], [30.0, 20.0, 10.0, 5.0], label="Test region", photon_energy_eV=1486.68),
            sigma_eV=0.5,
        ),
        observed_energy_eV=1.0,
        reference_energy_eV=1.2,
        reference_label="Test reference",
        uncertainty_eV=0.1,
    )
    destination = tmp_path / "project.xpsa"
    peaks = [
        {
            "name": "C 1s",
            "center": 1.5,
            "fwhm": 1.0,
            "area": 40.0,
            "profile": "doniach_sunjic",
            "eta": 0.0,
            "asymmetry": 0.12,
            "instrument_fwhm": 0.35,
        }
    ]
    ties = [{"target": "C 1s.eta", "source": "C 1s.eta", "scale": 1.0, "offset": 0.0}]

    settings = {"excluded_elements": ["O"], "rsf_overrides": [{"element": "C", "line": "1s", "value": 1.1}]}
    background_parameters = {"edge_fraction": 0.04, "max_iterations": 300, "tolerance": 1e-8}
    save_project(
        destination,
        spectrum=spectrum,
        background_method="shirley",
        background_parameters=background_parameters,
        peaks=peaks,
        ties=ties,
        analysis_settings=settings,
    )
    loaded = load_project(destination)

    assert loaded.spectrum.label == "Test region | Gaussian σ=0.5 eV"
    assert np.allclose(loaded.spectrum.energy, [0.2, 1.2, 2.2, 3.2])
    assert np.allclose(loaded.spectrum.original_energy, [0.0, 1.0, 2.0, 3.0])
    assert loaded.spectrum.photon_energy_eV == 1486.68
    assert len(loaded.spectrum.correction_history) == 1
    assert len(loaded.spectrum.processing_history) == 1
    assert loaded.analysis_settings == settings
    assert loaded.background_parameters == background_parameters
    assert loaded.background_method.value == "shirley"
    assert loaded.peaks == tuple(peaks)
    assert loaded.ties == tuple(ties)
    assert len(loaded.workspace) == 1


def test_complete_workspace_round_trip_preserves_retained_results(tmp_path) -> None:
    first = Spectrum([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 15.0, 9.0], label="Region A")
    second = Spectrum(
        [190.0, 191.0, 192.0, 193.0],
        [4.0, 5.0, 13.0, 4.5],
        label="B survey",
        photon_energy_eV=1486.68,
    )
    fit = FitResult(
        success=True,
        message="converged",
        parameter_values={"Peak 1.area": 12.5, "Peak 1.center": 2.1},
        standard_errors={"Peak 1.area": np.nan, "Peak 1.center": 0.02},
        background=np.array([5.0, 5.0, 5.0, 5.0]),
        components={"Peak 1": np.array([1.0, 4.0, 2.0, 0.5])},
        fitted_intensity=np.array([6.0, 9.0, 7.0, 5.5]),
        residual=np.array([0.1, -0.2, 0.05, 0.0]),
        weighted_residual=np.array([0.2, -0.4, 0.1, 0.0]),
        chi_square=0.21,
        reduced_chi_square=0.105,
        degrees_of_freedom=2,
        optimizer_evaluations=73,
        weighting="poisson",
        count_scale=1000.0,
        residual_rms=0.115,
        statistical_chi_square=0.21,
        reduced_statistical_chi_square=0.105,
        uncertainty_model="Pearson Poisson",
    )
    detected = DetectedPeak(192.0, 192.05, 13.0, 8.0, 1.1, 9.4, 6.5, targeted=True)
    reference = ReferenceLine("B", "Boron", "1s", 191.5, screening_context="high-BE environment")
    match = SurveyMatch(reference, detected, 191.5, 0.5, 81.0, ("Cl",))
    secondary = ExpectedSecondaryLine(reference, 191.5, True, 0.5)
    candidate = ElementCandidate("B", "Boron", 81.0, 1.0, (match,), (secondary,), ("Cl",))
    survey = SurveyAnalysis(
        spectrum_label=second.label,
        energy_range=second.energy_range,
        energy_scale=EnergyScale.BINDING,
        photon_energy_eV=1486.68,
        tolerance=2.5,
        noise=0.2,
        detection_limit=0.6,
        peaks=(detected,),
        matches=(match,),
        candidates=(candidate,),
        unassigned_peaks=(),
        included_elements=("B", "Cl"),
        excluded_elements=("O",),
        trace_searched_elements=("B",),
        trace_recovered_elements=("B",),
        trace_minimum_snr=2.5,
    )
    peak_row = {
        "name": "Peak 1",
        "center": 2.1,
        "fwhm": 1.2,
        "area": 12.5,
        "profile": "pseudo_voigt",
        "eta": 0.4,
    }
    stability = StabilityConsensusResult(
        spectrum_label=first.label,
        requested_peak_count=1,
        peak_count_probabilities={1: 0.92, 2: 0.08},
        bic_peak_count_probabilities={1: 0.94, 2: 0.06},
        aicc_peak_count_probabilities={1: 0.90, 2: 0.10},
        bic_by_peak_count={1: 12.0, 2: 16.9},
        aicc_by_peak_count={1: 10.5, 2: 15.2},
        components=(
            ComponentStability(
                "Consensus Peak 1",
                18,
                0.94,
                0.91,
                2.10,
                0.04,
                1.20,
                0.06,
                12.5,
                0.8,
                0.064,
                0.10,
                "LOW",
                0.20,
                "MODERATE",
                0.02,
                0.0,
                0.05,
                0.48,
                88.0,
                "STRONG NUMERICAL SUPPORT",
            ),
        ),
        requested_runs=12,
        successful_runs=34,
        failed_runs=2,
        paired_model_scenarios=11,
        candidate_trials_per_run=4,
        random_seed=73109,
        background_models=("Linear", "Shirley"),
        profiles=("gaussian", "pseudo_voigt"),
        weighting="poisson",
        count_scale=1000.0,
        calibration_uncertainty_eV=0.1,
        roi_jitter_fraction=0.025,
        bootstrap_strength=1.0,
        calibration_sigma_scale=1.0,
        max_nfev=900,
        max_background_above_fraction=0.12,
        numerical_note="Trust scores do not prove chemistry.",
    )
    items = (
        WorkspaceItemData(
            first,
            BackgroundMethod.CUSTOM,
            {"values": np.array([5.0, 5.0, 5.0, 5.0]), "source": "audited trace"},
            (peak_row,),
            (),
            fit,
            None,
            stability,
        ),
        WorkspaceItemData(
            second,
            BackgroundMethod.SHIRLEY,
            {"edge_fraction": 0.04, "max_iterations": 250},
            (),
            (),
            None,
            survey,
        ),
    )
    destination = tmp_path / "complete-workspace.xpsa"
    ui_state = {"workflow_tab": 1, "workspace_visible": True, "survey_visible": True, "quant_visible": False}
    settings = {"selected_elements": ["B", "Cl"], "trace_recovery_enabled": True}

    save_project(
        destination,
        spectrum=second,
        background_method="shirley",
        background_parameters=items[1].background_parameters,
        peaks=(),
        analysis_settings=settings,
        workspace_items=items,
        active_workspace_index=1,
        ui_state=ui_state,
    )
    loaded = load_project(destination)

    assert loaded.format_version == PROJECT_FORMAT_VERSION
    assert loaded.active_workspace_index == 1
    assert loaded.ui_state == ui_state
    assert [item.spectrum.label for item in loaded.workspace] == ["Region A", "B survey"]
    assert np.allclose(loaded.workspace[0].background_parameters["values"], [5.0] * 4)
    retained_fit = loaded.workspace[0].fit_result
    assert retained_fit is not None
    assert retained_fit.optimizer_evaluations == 73
    assert retained_fit.has_calibrated_chi_square
    assert np.isnan(retained_fit.standard_errors["Peak 1.area"])
    assert np.allclose(retained_fit.components["Peak 1"], [1.0, 4.0, 2.0, 0.5])
    retained_stability = loaded.workspace[0].consensus_result
    assert retained_stability is not None
    assert retained_stability.preferred_peak_count == 1
    assert retained_stability.bic_peak_count_probabilities[1] == 0.94
    assert retained_stability.aicc_peak_count_probabilities[1] == 0.90
    assert retained_stability.components[0].trust_score == 88.0
    assert retained_stability.components[0].identifiability == "STRONG NUMERICAL SUPPORT"
    retained_survey = loaded.workspace[1].survey_analysis
    assert retained_survey is not None
    assert retained_survey.trace_recovered_elements == ("B",)
    assert retained_survey.candidates[0].matches[0].reference.screening_context == "high-BE environment"

    corrupt_payload = json.loads(destination.read_text(encoding="utf-8"))
    corrupt_payload["workspace"]["items"][0]["fit_result"]["background"] = [5.0]
    corrupt = tmp_path / "misaligned-fit.xpsa"
    corrupt.write_text(json.dumps(corrupt_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="workspace content"):
        load_project(corrupt)


def test_version_five_project_migrates_to_single_workspace(tmp_path) -> None:
    destination = tmp_path / "legacy-v5.xpsa"
    spectrum = Spectrum([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 2.0], label="Legacy")
    save_project(destination, spectrum=spectrum, background_method="linear", peaks=())
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["version"] = 5
    payload.pop("workspace")
    payload.pop("ui")
    destination.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_project(destination)

    assert loaded.format_version == 5
    assert loaded.active_workspace_index == 0
    assert len(loaded.workspace) == 1
    assert loaded.workspace[0].spectrum.label == "Legacy"


def test_project_round_trip_preserves_complete_model_comparison(tmp_path) -> None:
    energy = np.linspace(280.0, 288.0, 120)
    background = np.linspace(3.0, 3.2, energy.size)
    intensity = background.copy()
    intensity += evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=282.5, fwhm=0.8, area=80.0)
    intensity += evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=285.5, fwhm=1.0, area=110.0)
    spectrum = Spectrum(energy, intensity, label="Retained alternatives")
    first = PeakComponent(
        "A",
        282.4,
        0.9,
        75.0,
        PeakProfile.GAUSSIAN,
        center_bounds=(281.5, 283.5),
        fwhm_bounds=(0.2, 2.0),
        area_bounds=(0.0, 250.0),
    )
    second = PeakComponent(
        "B",
        285.4,
        1.1,
        100.0,
        PeakProfile.GAUSSIAN,
        center_bounds=(284.5, 286.5),
        fwhm_bounds=(0.2, 2.0),
        area_bounds=(0.0, 250.0),
    )
    retained_background = BackgroundSettings(BackgroundMethod.CUSTOM, {"values": background.tolist()})
    comparison = compare_models(
        spectrum,
        (
            ModelCandidate("k2", "Two peaks", (first, second), (), retained_background, "Two-peak hypothesis."),
            ModelCandidate("k1", "One peak", (first,), (), retained_background, "One-peak hypothesis."),
        ),
        ModelComparisonSettings(
            compare_backgrounds=False,
            compare_profiles=False,
            compare_neighbor_counts=False,
            max_nfev=500,
        ),
    )
    destination = tmp_path / "comparison-project.xpsa"
    item = WorkspaceItemData(
        spectrum,
        BackgroundMethod.CUSTOM,
        {"values": background.tolist()},
        model_comparison=comparison,
    )

    save_project(
        destination,
        spectrum=spectrum,
        background_method=BackgroundMethod.CUSTOM,
        background_parameters=item.background_parameters,
        peaks=(),
        workspace_items=(item,),
        active_workspace_index=0,
    )
    loaded = load_project(destination)
    restored = loaded.workspace[0].model_comparison

    assert restored is not None
    assert restored.recommended_candidate_id == "k2"
    assert [candidate.candidate.identifier for candidate in restored.candidates] == ["k2", "k1"]
    assert np.allclose(
        restored.recommended.fit_result.fitted_intensity,
        comparison.recommended.fit_result.fitted_intensity,
    )
    assert restored.recommended.max_parameter_correlation == pytest.approx(
        comparison.recommended.max_parameter_correlation
    )
    assert restored.recommended.candidate.background.parameters["values"] == pytest.approx(background)
