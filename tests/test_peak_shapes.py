import numpy as np
import pytest

from xps_afruz.fitting.peak_shapes import (
    PeakProfile,
    ProfileFamily,
    asymmetric_pseudo_voigt,
    doniach_sunjic,
    gaussian,
    gaussian_lorentzian_product,
    lorentzian,
    pseudo_voigt,
    voigt,
)


@pytest.mark.parametrize("function, kwargs", [
    (gaussian, {}),
    (lorentzian, {}),
    (pseudo_voigt, {"eta": 0.3}),
])
def test_profiles_are_area_normalized(function, kwargs) -> None:
    energy = np.linspace(-100.0, 100.0, 100_001)
    curve = function(energy, center=0.0, fwhm=1.0, area=12.0, **kwargs)

    # A Lorentzian has long tails beyond the finite integration window.
    assert np.trapezoid(curve, energy) == pytest.approx(12.0, rel=0.005)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (gaussian_lorentzian_product, {}),
        (voigt, {"eta": 0.35}),
        (asymmetric_pseudo_voigt, {"eta": 0.35, "asymmetry": 0.25}),
        (doniach_sunjic, {"asymmetry": 0.12, "instrument_fwhm": 0.4}),
    ],
)
def test_phase_six_finite_window_profiles_are_area_normalized(function, kwargs) -> None:
    energy = np.linspace(-30.0, 30.0, 20_001)
    curve = function(energy, center=0.0, fwhm=1.2, area=12.0, **kwargs)

    assert np.all(np.isfinite(curve))
    assert np.all(curve >= 0.0)
    assert np.trapezoid(curve, energy) == pytest.approx(12.0, rel=1e-8)


def test_doniach_sunjic_is_classified_and_has_a_low_binding_energy_tail() -> None:
    energy = np.linspace(-20.0, 20.0, 40_001)
    curve = doniach_sunjic(energy, center=0.0, fwhm=1.2, area=12.0, asymmetry=0.15)

    left_tail = np.trapezoid(curve[energy < -0.8], energy[energy < -0.8])
    right_tail = np.trapezoid(curve[energy > 0.8], energy[energy > 0.8])
    assert PeakProfile.DONIACH_SUNJIC.family is ProfileFamily.METALLIC_XPS
    assert PeakProfile.VOIGT.family is ProfileFamily.MATHEMATICAL
    assert left_tail > right_tail
