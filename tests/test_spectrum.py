import numpy as np
import pytest

from xps_afruz import Spectrum


def test_spectrum_sorts_descending_energy_with_intensity() -> None:
    spectrum = Spectrum([4.0, 3.0, 2.0, 1.0], [40.0, 30.0, 20.0, 10.0])

    assert np.allclose(spectrum.energy, [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(spectrum.intensity, [10.0, 20.0, 30.0, 40.0])


def test_region_requires_enough_samples() -> None:
    spectrum = Spectrum(np.arange(8.0), np.arange(8.0))

    subset = spectrum.region(2.0, 6.0)

    assert subset.size == 5
    with pytest.raises(ValueError, match="at least four"):
        spectrum.region(0.0, 1.0)


def test_duplicate_energy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Spectrum([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
