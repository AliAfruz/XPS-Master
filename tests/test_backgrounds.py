import numpy as np

from xps_afruz.backgrounds import (
    BackgroundSettings,
    TougaardVariant,
    custom_background,
    diagnose_background,
    linear_background,
    shirley_background,
    tougaard_background,
)
from xps_afruz.core.spectrum import Spectrum


def test_linear_background_respects_provided_anchor_levels() -> None:
    energy = np.linspace(0.0, 10.0, 100)
    intensity = 3.0 + 0.4 * energy

    background = linear_background(energy, intensity, left_level=3.0, right_level=7.0)

    assert np.allclose(background, intensity)


def test_shirley_background_is_finite_and_preserves_endpoints() -> None:
    energy = np.linspace(280.0, 290.0, 300)
    intensity = 8.0 + 0.2 * (energy - 280.0) + 100.0 * np.exp(-np.square((energy - 285.0) / 0.7))

    background = shirley_background(energy, intensity, left_level=8.0, right_level=10.0)

    assert np.all(np.isfinite(background))
    assert background[0] == 8.0
    assert background[-1] == 10.0


def test_tougaard_background_is_anchored_for_nonnegative_signal() -> None:
    energy = np.linspace(100.0, 110.0, 100)
    intensity = 2.0 + np.exp(-np.square((energy - 105.0) / 1.0))

    background = tougaard_background(energy, intensity)

    assert np.all(background >= 0.0)
    assert np.isclose(background[0], 2.0)
    assert np.isclose(background[-1], 2.0)


def test_tougaard_accumulates_loss_tail_toward_higher_ascending_energy() -> None:
    energy = np.linspace(0.0, 10.0, 400)
    intensity = 1.0 + 30.0 * np.exp(-np.square((energy - 3.0) / 0.18))

    background = tougaard_background(energy, intensity, left_level=1.0, right_level=2.0)

    before_peak = int(np.argmin(np.abs(energy - 1.5)))
    after_peak = int(np.argmin(np.abs(energy - 6.0)))
    assert np.isclose(background[0], 1.0)
    assert np.isclose(background[-1], 2.0)
    assert background[after_peak] > background[before_peak]


def test_tougaard_automatic_scaling_honours_decreasing_edge_anchors() -> None:
    energy = np.linspace(0.0, 10.0, 400)
    intensity = 5.0 + 30.0 * np.exp(-np.square((energy - 3.0) / 0.18))

    background = tougaard_background(energy, intensity, left_level=5.0, right_level=3.0)

    assert np.isclose(background[0], 5.0)
    assert np.isclose(background[-1], 3.0)


def test_tougaard_u3_and_custom_background_are_explicit() -> None:
    energy = np.linspace(100.0, 110.0, 100)
    intensity = 2.0 + np.exp(-np.square((energy - 105.0) / 1.0))
    u3 = tougaard_background(energy, intensity, variant=TougaardVariant.U3, d=250.0)
    supplied = custom_background(energy, intensity, values=np.linspace(2.0, 3.0, energy.size))
    settings = BackgroundSettings("tougaard", {"variant": "u3", "b": 2866.0, "c": 1643.0, "d": 250.0})

    assert np.all(np.isfinite(u3))
    assert np.allclose(supplied, np.linspace(2.0, 3.0, energy.size))
    assert settings.as_dict()["parameters"]["variant"] == "u3"


def test_background_diagnostics_flags_material_over_background() -> None:
    energy = np.linspace(280.0, 290.0, 200)
    intensity = 5.0 + 20.0 * np.exp(-np.square((energy - 285.0) / 0.5))
    spectrum = Spectrum(energy, intensity)

    diagnostics = diagnose_background(spectrum, intensity + 10.0)

    assert diagnostics.is_cautionary
    assert diagnostics.fraction_above_signal == 1.0
    assert np.isclose(diagnostics.minimum_net_intensity, -10.0)
