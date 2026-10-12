import numpy as np
import pytest

from xps_afruz.chemistry.survey import analyze_survey, detect_survey_peaks
from xps_afruz.core.spectrum import Spectrum


def _gaussian(x: np.ndarray, center: float, height: float, fwhm: float = 1.4) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * np.square((x - center) / fwhm))


def test_survey_analysis_finds_known_reference_lines() -> None:
    energy = np.linspace(0.0, 1_100.0, 5_501)
    rng = np.random.default_rng(12)
    intensity = 25.0 + 0.015 * energy + rng.normal(0.0, 1.2, energy.size)
    intensity += _gaussian(energy, 284.8, 500.0)
    intensity += _gaussian(energy, 531.0, 760.0)
    intensity += _gaussian(energy, 710.8, 420.0)
    spectrum = Spectrum(energy, intensity, label="Known survey")

    analysis = analyze_survey(spectrum, tolerance=1.0)

    assert {candidate.symbol for candidate in analysis.candidates} >= {"C", "O", "Fe"}
    assert any(match.reference.label == "O 1s" and abs(match.energy_error) < 0.25 for match in analysis.matches)
    assert all(peak.prominence > 0.0 for peak in analysis.peaks)


def test_survey_peak_parameters_are_validated() -> None:
    spectrum = Spectrum(np.arange(20.0), np.ones(20))

    with pytest.raises(ValueError, match="Smoothing"):
        detect_survey_peaks(spectrum, smoothing_eV=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        analyze_survey(spectrum, tolerance=0.0)


def test_survey_search_can_be_restricted_to_selected_elements() -> None:
    energy = np.linspace(0.0, 1_200.0, 2_401)
    intensity = np.full_like(energy, 5.0)
    intensity += 250.0 * np.exp(-0.5 * ((energy - 284.8) / 1.5) ** 2)
    intensity += 300.0 * np.exp(-0.5 * ((energy - 531.0) / 1.8) ** 2)
    spectrum = Spectrum(energy, intensity, label="Selected element scope")

    analysis = analyze_survey(spectrum, tolerance=1.0, include_elements=("O",))

    assert analysis.included_elements == ("O",)
    assert {candidate.symbol for candidate in analysis.candidates} == {"O"}
    assert all(match.reference.symbol == "O" for match in analysis.matches)


def test_targeted_trace_search_recovers_a_weak_selected_line() -> None:
    energy = np.linspace(0.0, 600.0, 6_001)
    rng = np.random.default_rng(42)
    intensity = 20.0 + 0.01 * energy + rng.normal(0.0, 1.0, energy.size)
    intensity += _gaussian(energy, 284.8, 300.0, 2.0)
    intensity += _gaussian(energy, 531.0, 4.0, 2.0)
    spectrum = Spectrum(energy, intensity, label="Weak selected oxygen")

    ordinary = analyze_survey(
        spectrum,
        tolerance=1.0,
        include_elements=("C", "O"),
        prominence_sigma=6.0,
    )
    targeted = analyze_survey(
        spectrum,
        tolerance=1.0,
        include_elements=("C", "O"),
        target_elements=("O",),
        trace_minimum_snr=2.5,
        prominence_sigma=6.0,
    )

    assert "O" not in {candidate.symbol for candidate in ordinary.candidates}
    oxygen = next(candidate for candidate in targeted.candidates if candidate.symbol == "O")
    assert oxygen.confidence_label == "targeted trace"
    assert oxygen.matches[0].peak.targeted
    assert oxygen.matches[0].peak.signal_to_noise >= 2.5
    assert targeted.trace_searched_elements == ("O",)
    assert targeted.trace_recovered_elements == ("O",)


def test_targeted_boron_search_supports_high_binding_energy_b1s() -> None:
    energy = np.linspace(0.0, 300.0, 3_001)
    rng = np.random.default_rng(18)
    intensity = 20.0 + rng.normal(0.0, 1.0, energy.size)
    intensity += _gaussian(energy, 284.8, 300.0, 2.0)
    intensity += _gaussian(energy, 191.5, 4.5, 2.2)
    spectrum = Spectrum(energy, intensity, label="Weak high-BE B 1s")

    analysis = analyze_survey(
        spectrum,
        tolerance=2.5,
        include_elements=("B",),
        target_elements=("B",),
        trace_minimum_snr=2.5,
        prominence_sigma=6.0,
    )

    boron = next(candidate for candidate in analysis.candidates if candidate.symbol == "B")
    match = boron.matches[0]
    assert match.peak.targeted
    assert match.reference.line == "1s"
    assert match.reference.screening_context == "high-BE screening"
    assert abs(match.reference_energy - 191.5) < 1e-12
    assert boron.expected_secondary_lines == ()
