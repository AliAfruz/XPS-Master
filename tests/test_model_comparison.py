import numpy as np
import pytest

from xps_afruz import BackgroundMethod, BackgroundSettings, PeakProfile, Spectrum
from xps_afruz.fitting.constraint_templates import SPIN_ORBIT_TEMPLATES, spin_orbit_template
from xps_afruz.fitting.model import PeakComponent
from xps_afruz.fitting.model_comparison import (
    ModelCandidate,
    ModelComparisonCancelled,
    ModelComparisonSettings,
    build_standard_candidates,
    compare_models,
    optimized_components,
    shift_model_comparison_energy,
)
from xps_afruz.fitting.peak_shapes import evaluate_peak


def _doublet() -> tuple[Spectrum, np.ndarray, tuple[PeakComponent, ...]]:
    energy = np.linspace(280.0, 290.0, 260)
    background = 6.0 + 0.04 * (energy - energy[0])
    signal = background.copy()
    signal += evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=283.4, fwhm=0.85, area=100.0)
    signal += evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=286.5, fwhm=1.10, area=150.0)
    spectrum = Spectrum(energy, signal, label="Comparison doublet")
    components = (
        PeakComponent(
            "Peak A",
            283.2,
            1.0,
            90.0,
            PeakProfile.GAUSSIAN,
            center_bounds=(282.5, 284.2),
            fwhm_bounds=(0.3, 2.0),
            area_bounds=(0.0, 300.0),
        ),
        PeakComponent(
            "Peak B",
            286.7,
            1.0,
            130.0,
            PeakProfile.GAUSSIAN,
            center_bounds=(285.5, 287.5),
            fwhm_bounds=(0.3, 2.0),
            area_bounds=(0.0, 300.0),
        ),
    )
    return spectrum, background, components


def test_generic_spin_orbit_templates_expose_degeneracy_without_inventing_splitting() -> None:
    assert {template.key for template in SPIN_ORBIT_TEMPLATES} == {"custom", "p", "d", "f"}
    assert spin_orbit_template("p").upper_to_lower_area_ratio == pytest.approx(0.5)
    assert spin_orbit_template("d").upper_to_lower_area_ratio == pytest.approx(2.0 / 3.0)
    assert spin_orbit_template("f").upper_to_lower_area_ratio == pytest.approx(0.75)
    assert all("splitting" in template.note.lower() or "deviations" in template.note.lower() for template in SPIN_ORBIT_TEMPLATES)


def test_standard_candidate_family_varies_one_documented_choice_at_a_time() -> None:
    spectrum, background, components = _doublet()
    settings = ModelComparisonSettings(max_nfev=400)
    candidates = build_standard_candidates(
        spectrum,
        components,
        (),
        BackgroundSettings(BackgroundMethod.CUSTOM, {"values": background.tolist()}),
        settings,
    )

    assert candidates[0].label == "Analyst model"
    assert len({candidate.identifier for candidate in candidates}) == len(candidates)
    assert {candidate.background.method for candidate in candidates} >= {
        BackgroundMethod.CUSTOM,
        BackgroundMethod.LINEAR,
        BackgroundMethod.SHIRLEY,
        BackgroundMethod.TOUGAARD,
    }
    assert {len(candidate.components) for candidate in candidates} == {1, 2, 3}
    assert any({component.profile for component in candidate.components} == {PeakProfile.PSEUDO_VOIGT} for candidate in candidates)


def test_comparison_prefers_supported_doublet_and_retains_complete_alternatives() -> None:
    spectrum, background, components = _doublet()
    background_settings = BackgroundSettings(BackgroundMethod.CUSTOM, {"values": background.tolist()})
    candidates = (
        ModelCandidate("k2", "Two components", components, (), background_settings, "Known two-component hypothesis."),
        ModelCandidate("k1", "One component", (components[0],), (), background_settings, "Parsimonious one-component hypothesis."),
    )
    progress: list[tuple[int, int]] = []
    result = compare_models(
        spectrum,
        candidates,
        ModelComparisonSettings(compare_backgrounds=False, compare_profiles=False, compare_neighbor_counts=False, max_nfev=800),
        progress_callback=lambda completed, total, _message: progress.append((completed, total)),
    )

    assert result.recommended_candidate_id == "k2"
    assert result.recommended.rank == 1
    assert result.recommended.fit_result.residual_rms < result.candidate("k1").fit_result.residual_rms
    assert sum(candidate.aicc_weight for candidate in result.candidates) == pytest.approx(1.0)
    assert sum(candidate.bic_weight for candidate in result.candidates) == pytest.approx(1.0)
    assert len(optimized_components(result.recommended)) == 2
    assert progress[-1] == (2, 2)
    assert "not a chemical-state" in result.numerical_note


def test_comparison_cancellation_and_energy_shift_are_non_destructive() -> None:
    spectrum, background, components = _doublet()
    background_settings = BackgroundSettings(BackgroundMethod.CUSTOM, {"values": background.tolist()})
    candidates = (
        ModelCandidate("k2", "Two components", components, (), background_settings, "Two components."),
        ModelCandidate("k1", "One component", (components[0],), (), background_settings, "One component."),
    )
    settings = ModelComparisonSettings(compare_backgrounds=False, compare_profiles=False, compare_neighbor_counts=False, max_nfev=400)
    with pytest.raises(ModelComparisonCancelled):
        compare_models(spectrum, candidates, settings, cancellation_callback=lambda: True)

    result = compare_models(spectrum, candidates, settings)
    shifted = shift_model_comparison_energy(result, 1.25)
    assert shifted.recommended.fit_result.parameter_values["Peak A.center"] == pytest.approx(
        result.recommended.fit_result.parameter_values["Peak A.center"] + 1.25
    )
    assert np.array_equal(shifted.recommended.fit_result.fitted_intensity, result.recommended.fit_result.fitted_intensity)
    assert result.recommended.candidate.components[0].center == components[0].center
