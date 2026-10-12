import numpy as np
import pytest

from xps_afruz import FitModel, PeakComponent, PeakProfile, Spectrum, fit_spectrum, make_spin_orbit_doublet
from xps_afruz.fitting.peak_shapes import evaluate_peak
from xps_afruz.fitting.model import shift_fit_result_energy, shift_parameter_ties_energy


def test_bounded_fit_recovers_a_single_gaussian() -> None:
    energy = np.linspace(280.0, 290.0, 600)
    background = 9.0 + 0.15 * (energy - 280.0)
    observed = background + evaluate_peak(
        energy, profile=PeakProfile.GAUSSIAN, center=284.75, fwhm=1.15, area=300.0
    )
    spectrum = Spectrum(energy, observed)
    model = FitModel(
        [
            PeakComponent(
                "C 1s",
                center=284.2,
                fwhm=1.5,
                area=240.0,
                profile=PeakProfile.GAUSSIAN,
                center_bounds=(283.5, 286.0),
                fwhm_bounds=(0.5, 2.5),
                area_bounds=(1.0, 500.0),
            )
        ],
        background,
    )

    result = fit_spectrum(spectrum, model)

    assert result.success
    assert result.parameter_values["C 1s.center"] == pytest.approx(284.75, abs=1e-4)
    assert result.parameter_values["C 1s.fwhm"] == pytest.approx(1.15, abs=1e-4)
    assert result.parameter_values["C 1s.area"] == pytest.approx(300.0, abs=1e-3)
    assert result.chi_square == pytest.approx(0.0, abs=1e-7)
    assert result.statistical_chi_square is None
    assert result.residual_rms == pytest.approx(0.0, abs=1e-7)


def test_poisson_weighting_exposes_calibrated_statistical_chi_square() -> None:
    energy = np.linspace(280.0, 290.0, 400)
    background = np.full_like(energy, 30.0)
    observed = background + evaluate_peak(energy, profile=PeakProfile.GAUSSIAN, center=284.8, fwhm=1.1, area=300.0)
    model = FitModel(
        [
            PeakComponent(
                "C 1s",
                center=284.8,
                fwhm=1.1,
                area=300.0,
                profile=PeakProfile.GAUSSIAN,
                center_bounds=(283.0, 286.0),
                fwhm_bounds=(0.5, 2.0),
                area_bounds=(1.0, 500.0),
            )
        ],
        background,
    )

    result = fit_spectrum(Spectrum(energy, observed), model, weighting="poisson", count_scale=1.0)

    assert result.has_calibrated_chi_square
    assert result.statistical_chi_square == pytest.approx(0.0, abs=1e-8)
    assert result.reduced_statistical_chi_square == pytest.approx(0.0, abs=1e-8)
    assert "Poisson" in result.uncertainty_model


def test_spin_orbit_ties_are_applied_during_fit() -> None:
    energy = np.linspace(150.0, 160.0, 700)
    background = np.full_like(energy, 5.0)
    observed = background.copy()
    observed += evaluate_peak(energy, profile="gaussian", center=153.4, fwhm=1.0, area=400.0)
    observed += evaluate_peak(energy, profile="gaussian", center=155.0, fwhm=1.0, area=200.0)
    components, ties = make_spin_orbit_doublet(
        "Test 3d",
        center=153.0,
        fwhm=1.3,
        area=300.0,
        splitting=1.6,
        upper_to_lower_area_ratio=0.5,
        profile="gaussian",
        center_bounds=(152.0, 154.0),
        fwhm_bounds=(0.5, 2.0),
        area_bounds=(1.0, 700.0),
    )

    result = fit_spectrum(Spectrum(energy, observed), FitModel(components, background, ties))

    assert result.success
    assert result.parameter_values["Test 3d lower.center"] == pytest.approx(153.4, abs=1e-4)
    assert result.parameter_values["Test 3d upper.center"] == pytest.approx(155.0, abs=1e-4)
    assert result.parameter_values["Test 3d upper.area"] == pytest.approx(
        0.5 * result.parameter_values["Test 3d lower.area"], abs=1e-9
    )

    shifted = shift_fit_result_energy(result, 1.25)
    shifted_ties = shift_parameter_ties_energy(ties, 1.25)
    assert shifted.parameter_values["Test 3d lower.center"] == pytest.approx(154.65, abs=1e-4)
    assert shifted.parameter_values["Test 3d upper.center"] == pytest.approx(156.25, abs=1e-4)
    assert shifted.parameter_values["Test 3d lower.area"] == result.parameter_values["Test 3d lower.area"]
    assert np.array_equal(shifted.fitted_intensity, result.fitted_intensity)
    assert next(tie for tie in shifted_ties if tie.target.endswith("upper.center")).offset == pytest.approx(1.6)


def test_asymmetric_profile_parameter_is_optimized() -> None:
    energy = np.linspace(280.0, 290.0, 800)
    observed = evaluate_peak(
        energy,
        profile=PeakProfile.ASYMMETRIC_PSEUDO_VOIGT,
        center=284.8,
        fwhm=1.2,
        area=300.0,
        eta=0.3,
        asymmetry=0.28,
    )
    model = FitModel(
        [
            PeakComponent(
                "Asymmetric C 1s",
                center=284.5,
                fwhm=1.5,
                area=250.0,
                profile=PeakProfile.ASYMMETRIC_PSEUDO_VOIGT,
                eta=0.3,
                asymmetry=0.1,
                center_bounds=(283.0, 286.0),
                fwhm_bounds=(0.5, 2.5),
                area_bounds=(1.0, 500.0),
                vary_asymmetry=True,
            )
        ],
        np.zeros_like(energy),
    )

    result = fit_spectrum(Spectrum(energy, observed), model)

    assert result.success
    assert result.parameter_values["Asymmetric C 1s.asymmetry"] == pytest.approx(0.28, abs=1e-4)
    assert result.parameter_values["Asymmetric C 1s.instrument_fwhm"] == 0.0
