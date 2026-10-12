import numpy as np

from xps_afruz.chemistry.quantification import (
    GENERIC_ALKA_RSF,
    compare_compositions,
    load_sensitivity_factor_library,
    quantify_survey,
    save_sensitivity_factor_library,
)
from xps_afruz.chemistry.survey import analyze_survey
from xps_afruz.core.spectrum import Spectrum


def _peak(energy: np.ndarray, center: float, height: float, fwhm: float = 1.5) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * np.square((energy - center) / fwhm))


def test_quantification_normalizes_selected_rsf_corrected_areas() -> None:
    energy = np.linspace(0.0, 1_100.0, 11_001)
    intensity = 20.0 + _peak(energy, 284.8, 500.0) + _peak(energy, 531.0, 700.0)
    spectrum = Spectrum(energy, intensity, photon_energy_eV=1486.68)
    analysis = analyze_survey(spectrum, tolerance=0.8)
    result = quantify_survey(analysis)
    overridden = quantify_survey(analysis, factor_overrides={("O", "1s"): 1.0})

    assert {record.symbol for record in result.records} == {"C", "O"}
    assert np.isclose(sum(record.atomic_percent for record in result.records), 100.0)
    assert overridden.as_percentages()["O"] > result.as_percentages()["O"]


def test_survey_consistency_auger_and_exclusion_are_explicit() -> None:
    energy = np.linspace(0.0, 1_100.0, 11_001)
    intensity = 15.0 + _peak(energy, 531.0, 650.0) + _peak(energy, 976.68, 360.0)
    spectrum = Spectrum(energy, intensity, photon_energy_eV=1486.68)
    analysis = analyze_survey(spectrum, tolerance=0.8)
    oxygen = next(candidate for candidate in analysis.candidates if candidate.symbol == "O")
    excluded = analyze_survey(spectrum, tolerance=0.8, exclude_elements=["O"])

    assert any(match.reference.feature_type == "auger" for match in oxygen.matches)
    assert oxygen.consistency_score > 0.0
    assert "O" not in {candidate.symbol for candidate in excluded.candidates}
    comparison = compare_compositions({"sample A": quantify_survey(analysis), "sample B": quantify_survey(analysis)})
    assert comparison.values["O"]["sample A"] > 0.0


def test_targeted_trace_is_quantified_with_detection_uncertainty() -> None:
    energy = np.linspace(0.0, 600.0, 6_001)
    rng = np.random.default_rng(42)
    intensity = 20.0 + 0.01 * energy + rng.normal(0.0, 1.0, energy.size)
    intensity += _peak(energy, 284.8, 300.0, 2.0)
    intensity += _peak(energy, 531.0, 4.0, 2.0)
    analysis = analyze_survey(
        Spectrum(energy, intensity),
        tolerance=1.0,
        include_elements=("C", "O"),
        target_elements=("O",),
        trace_minimum_snr=2.5,
        prominence_sigma=6.0,
    )

    result = quantify_survey(analysis)
    oxygen = next(record for record in result.records if record.symbol == "O")

    assert oxygen.targeted_trace
    assert 0.0 < oxygen.atomic_percent < 2.0
    assert oxygen.atomic_percent_uncertainty > 0.25 * oxygen.atomic_percent


def test_user_defined_rsf_quantifies_a_line_missing_from_generic_library() -> None:
    energy = np.linspace(20.0, 120.0, 2_001)
    intensity = 10.0 + _peak(energy, 84.0, 120.0, 1.8)
    analysis = analyze_survey(
        Spectrum(energy, intensity),
        tolerance=0.8,
        include_elements=("Au",),
    )

    missing = quantify_survey(analysis)
    supplied = quantify_survey(analysis, factor_overrides={("Au", "4f7/2"): 7.5})

    assert missing.unquantified_elements == ("Au",)
    assert supplied.as_percentages() == {"Au": 100.0}
    assert supplied.records[0].sensitivity_factor.model == "user_defined_rsf"


def test_instrument_rsf_library_round_trip(tmp_path) -> None:
    destination = tmp_path / "laboratory-rsf.json"

    save_sensitivity_factor_library(destination, GENERIC_ALKA_RSF[:2])
    loaded = load_sensitivity_factor_library(destination)

    assert [(factor.symbol, factor.line, factor.value) for factor in loaded] == [("C", "1s", 1.0), ("N", "1s", 1.8)]
