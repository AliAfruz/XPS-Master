import numpy as np

from xps_afruz import Spectrum
from xps_afruz.core.processing import (
    combine_spectra,
    derivative_spectrum,
    despike_spectrum,
    gaussian_smooth,
    normalize_spectrum,
    savitzky_golay_smooth,
    signal_to_noise,
    subtract_spectra,
)


def test_processing_operations_are_non_destructive_and_recorded() -> None:
    energy = np.linspace(0.0, 10.0, 201)
    intensity = 5.0 + np.exp(-np.square((energy - 5.0) / 0.4))
    spectrum = Spectrum(energy, intensity)

    normalized = normalize_spectrum(spectrum, mode="maximum")
    smooth = savitzky_golay_smooth(normalized, window_eV=0.5, polynomial_order=2)
    derivative = derivative_spectrum(smooth, order=1)

    assert np.allclose(spectrum.intensity, intensity)
    assert np.isclose(np.max(normalized.intensity), 1.0)
    assert [entry["operation"] for entry in derivative.processing_history] == ["normalization", "savitzky_golay", "derivative"]
    assert derivative.intensity_unit.endswith("/ eV^1")


def test_despike_and_common_grid_spectrum_math() -> None:
    energy = np.linspace(0.0, 10.0, 101)
    baseline = 3.0 + 0.2 * energy
    spiky = baseline.copy()
    spiky[50] += 200.0
    spectrum = Spectrum(energy, spiky, label="spiky")
    cleaned, detection = despike_spectrum(spectrum, threshold_sigma=4.0)
    other = Spectrum(energy[5:-5], baseline[5:-5], label="baseline")
    added = combine_spectra((cleaned, other), operation="add")
    difference = subtract_spectra(cleaned, other)

    assert 50 in detection.indices
    assert cleaned.intensity[50] < spectrum.intensity[50]
    assert added.size == difference.size
    assert difference.processing_history[-1]["operation"] == "spectrum_subtraction"


def test_gaussian_smoothing_and_snr_return_finite_values() -> None:
    energy = np.linspace(0.0, 4.0, 81)
    rng = np.random.default_rng(7)
    spectrum = Spectrum(energy, 2.0 + np.sin(energy) + rng.normal(0.0, 0.1, energy.size))

    smoothed = gaussian_smooth(spectrum, sigma_eV=0.1)
    metrics = signal_to_noise(smoothed)

    assert np.all(np.isfinite(smoothed.intensity))
    assert metrics.noise > 0.0
    assert metrics.signal_to_noise > 0.0
