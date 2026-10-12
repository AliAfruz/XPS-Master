import numpy as np
import pytest

from xps_afruz.core.energy import (
    EnergyCalibration,
    EnergyScale,
    apply_energy_calibration,
    calibration_comparison,
    charge_correct,
    convert_energy_scale,
)
from xps_afruz.core.spectrum import Spectrum


def test_energy_conversion_and_charge_correction_preserve_import_axis() -> None:
    spectrum = Spectrum(
        [280.0, 281.0, 282.0, 283.0],
        [10.0, 20.0, 30.0, 20.0],
        energy_scale=EnergyScale.BINDING,
        photon_energy_eV=1486.68,
    )
    corrected = charge_correct(
        spectrum,
        observed_energy_eV=282.0,
        reference_energy_eV=284.8,
        reference_label="C 1s",
        uncertainty_eV=0.2,
    )
    kinetic = convert_energy_scale(corrected, EnergyScale.KINETIC)

    assert np.allclose(corrected.original_energy, spectrum.original_energy)
    assert np.allclose(corrected.energy, spectrum.energy + 2.8)
    assert corrected.energy_uncertainty_eV == 0.2
    assert kinetic.energy_scale is EnergyScale.KINETIC
    assert kinetic.original_energy_scale is EnergyScale.BINDING
    assert len(kinetic.correction_history) == 2
    assert np.isclose(calibration_comparison(corrected)["median_shift_eV"], 2.8)


def test_kinetic_conversion_requires_recorded_or_supplied_photon_energy() -> None:
    spectrum = Spectrum(np.arange(4.0), np.ones(4), energy_scale=EnergyScale.KINETIC)

    try:
        convert_energy_scale(spectrum, EnergyScale.BINDING)
    except ValueError as error:
        assert "photon energy" in str(error)
    else:
        raise AssertionError("Expected a missing-photon-energy validation error")


def test_workspace_calibration_records_scope_without_replacing_import_axis() -> None:
    spectrum = Spectrum(np.arange(4.0), np.ones(4))
    shifted = apply_energy_calibration(
        spectrum,
        EnergyCalibration(
            shift_eV=-0.75,
            reference_label="Au 4f project reference",
            uncertainty_eV=0.05,
            method="project_wide_manual_shift",
            scope="complete_workspace",
            affected_spectra=7,
        ),
    )

    assert np.allclose(shifted.energy, spectrum.energy - 0.75)
    assert np.allclose(shifted.original_energy, spectrum.original_energy)
    assert shifted.correction_history[-1]["scope"] == "complete_workspace"
    assert shifted.correction_history[-1]["affected_spectra"] == 7
    assert shifted.energy_uncertainty_eV == pytest.approx(0.05)
