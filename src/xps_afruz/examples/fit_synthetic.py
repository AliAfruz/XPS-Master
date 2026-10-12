"""Fit a noisy synthetic spin-orbit doublet with physical ties."""

from __future__ import annotations

import numpy as np

from xps_afruz import FitModel, PeakProfile, Spectrum, fit_spectrum, make_spin_orbit_doublet
from xps_afruz.fitting.peak_shapes import evaluate_peak


def synthetic_doublet(seed: int = 7) -> tuple[Spectrum, np.ndarray]:
    """Create a deterministic test spectrum and return its exact background."""

    energy = np.linspace(154.0, 171.0, 900)
    background = 38.0 + 0.45 * (energy - energy[0])
    signal = background.copy()
    signal += evaluate_peak(
        energy,
        profile=PeakProfile.PSEUDO_VOIGT,
        center=160.1,
        fwhm=1.25,
        area=900.0,
        eta=0.25,
    )
    signal += evaluate_peak(
        energy,
        profile=PeakProfile.PSEUDO_VOIGT,
        center=161.85,
        fwhm=1.25,
        area=450.0,
        eta=0.25,
    )
    noise = np.random.default_rng(seed).normal(0.0, 1.8, size=energy.size)
    return Spectrum(energy, signal + noise, label="Synthetic doublet"), background


def main() -> None:
    spectrum, background = synthetic_doublet()
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
    result = fit_spectrum(Spectrum(spectrum.energy, spectrum.intensity), FitModel(components, background, ties))
    print(f"Success: {result.success} ({result.message})")
    print(f"Residual RMS: {result.residual_rms:.4f} counts (uniform fit; chi-square is not statistically calibrated)")
    for key in ("Synthetic 3d lower.center", "Synthetic 3d lower.fwhm", "Synthetic 3d lower.area"):
        print(f"{key}: {result.parameter_values[key]:.4f} +/- {result.standard_errors[key]:.4f}")


if __name__ == "__main__":
    main()
