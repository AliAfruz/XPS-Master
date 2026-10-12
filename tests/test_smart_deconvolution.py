import numpy as np
import pytest

from xps_afruz import PeakProfile, SmartDeconvolutionSettings, Spectrum, smart_deconvolve
from xps_afruz.fitting.peak_shapes import evaluate_peak


def test_smart_deconvolution_recovers_requested_separated_components() -> None:
    energy = np.linspace(280.0, 290.0, 500)
    background = 3.0 + 0.03 * (energy - energy[0])
    observed = background.copy()
    observed += evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=283.6, fwhm=0.90, area=120.0)
    observed += evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=286.4, fwhm=1.15, area=180.0)

    result = smart_deconvolve(
        Spectrum(energy, observed),
        background,
        SmartDeconvolutionSettings(
            peak_count=2,
            profile=PeakProfile.GAUSSIAN,
            candidate_trials=16,
            max_nfev=1_000,
        ),
    )

    centers = sorted(value for name, value in result.fit_result.parameter_values.items() if name.endswith(".center"))
    assert len(result.components) == 2
    assert result.completed_trials == result.candidate_trials
    assert centers == pytest.approx([283.6, 286.4], abs=1e-3)
    assert result.fit_result.reduced_chi_square == pytest.approx(0.0, abs=1e-8)
