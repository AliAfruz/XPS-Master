import numpy as np
import pytest

from xps_afruz import Spectrum
from xps_afruz.backgrounds import BackgroundMethod
from xps_afruz.fitting.global_series import (
    GlobalSeriesCancelled,
    GlobalSeriesSettings,
    fit_global_series,
    shift_global_series_energy,
)
from xps_afruz.fitting.model import ParameterTie, PeakComponent
from xps_afruz.fitting.peak_shapes import PeakProfile, evaluate_peak
from xps_afruz.io.project_io import WorkspaceItemData, load_project, save_project


def _series() -> tuple[list[Spectrum], list[np.ndarray], list[PeakComponent]]:
    rng = np.random.default_rng(1234)
    energy = np.linspace(526.0, 536.0, 401)
    background = 3.0 + 0.025 * (energy - energy[0])
    shifts = (0.0, 0.12, -0.08)
    areas = (520.0, 790.0, 350.0)
    spectra = []
    for index, (shift, area) in enumerate(zip(shifts, areas, strict=True)):
        signal = evaluate_peak(
            energy,
            profile=PeakProfile.PSEUDO_VOIGT,
            center=531.1 + shift,
            fwhm=1.35,
            area=area,
            eta=0.32,
        )
        spectra.append(Spectrum(energy, background + signal + rng.normal(0.0, 0.12, energy.size), label=f"Scan {index + 1}"))
    components = [
        PeakComponent(
            "O 1s",
            center=531.0,
            fwhm=1.5,
            area=500.0,
            profile=PeakProfile.PSEUDO_VOIGT,
            eta=0.32,
            center_bounds=(529.5, 532.5),
            fwhm_bounds=(0.5, 3.0),
            area_bounds=(1.0, 2_000.0),
        )
    ]
    return spectra, [background.copy() for _ in spectra], components


def test_global_series_recovers_shared_shape_areas_and_offsets() -> None:
    spectra, backgrounds, components = _series()
    progress: list[int] = []
    result = fit_global_series(
        spectra,
        backgrounds,
        components,
        workspace_indices=(2, 4, 7),
        settings=GlobalSeriesSettings(maximum_energy_shift_eV=0.25, max_nfev=2_000),
        progress_callback=lambda percent, _message: progress.append(percent),
    )

    assert result.success
    assert result.workspace_indices == (2, 4, 7)
    assert result.global_parameter_count == 7
    assert result.independent_parameter_count == 9
    assert result.degrees_of_freedom == 3 * 401 - 7
    assert sum(item.global_fit.degrees_of_freedom for item in result.spectrum_fits) == result.degrees_of_freedom
    assert result.spectrum_fits[0].energy_shift_eV == 0.0
    assert result.spectrum_fits[1].energy_shift_eV == pytest.approx(0.12, abs=0.015)
    assert result.spectrum_fits[2].energy_shift_eV == pytest.approx(-0.08, abs=0.015)
    trend = result.component_trends[0]
    assert trend.reference_center_eV == pytest.approx(531.1, abs=0.015)
    assert trend.fwhm_eV == pytest.approx((1.35, 1.35, 1.35), abs=0.015)
    assert trend.areas == pytest.approx((520.0, 790.0, 350.0), rel=0.01)
    assert result.delta_bic > 0.0
    assert progress[-1] == 100
    for spectrum, fitted in zip(spectra, result.spectrum_fits, strict=True):
        assert fitted.global_fit.fitted_intensity.shape == spectrum.energy.shape
        assert fitted.global_fit.residual.shape == spectrum.energy.shape


def test_global_series_preserves_parameter_ties_in_every_spectrum() -> None:
    spectra, backgrounds, _ = _series()
    lower = PeakComponent(
        "Doublet lower",
        center=530.4,
        fwhm=1.2,
        area=300.0,
        center_bounds=(529.0, 533.0),
        fwhm_bounds=(0.5, 3.0),
        area_bounds=(1.0, 2_000.0),
    )
    upper = PeakComponent(
        "Doublet upper",
        center=531.1,
        fwhm=1.2,
        area=150.0,
        center_bounds=(529.7, 533.7),
        fwhm_bounds=(0.5, 3.0),
        area_bounds=(0.5, 1_000.0),
    )
    ties = (
        ParameterTie("Doublet upper.center", "Doublet lower.center", offset=0.7),
        ParameterTie("Doublet upper.fwhm", "Doublet lower.fwhm"),
        ParameterTie("Doublet upper.area", "Doublet lower.area", scale=0.5),
    )
    result = fit_global_series(
        spectra[:2],
        backgrounds[:2],
        (lower, upper),
        ties,
        settings=GlobalSeriesSettings(max_nfev=500),
    )
    for spectrum_fit in result.spectrum_fits:
        values = spectrum_fit.global_fit.parameter_values
        assert values["Doublet upper.center"] - values["Doublet lower.center"] == pytest.approx(0.7)
        assert values["Doublet upper.fwhm"] == pytest.approx(values["Doublet lower.fwhm"])
        assert values["Doublet upper.area"] == pytest.approx(values["Doublet lower.area"] * 0.5)


def test_global_series_cancel_and_uniform_energy_translation() -> None:
    spectra, backgrounds, components = _series()
    with pytest.raises(GlobalSeriesCancelled):
        fit_global_series(
            spectra,
            backgrounds,
            components,
            cancellation_callback=lambda: True,
        )

    result = fit_global_series(
        spectra[:2],
        backgrounds[:2],
        components,
        settings=GlobalSeriesSettings(max_nfev=1_000),
    )
    shifted = shift_global_series_energy(result, 0.45)
    assert shifted.component_trends[0].reference_center_eV == pytest.approx(
        result.component_trends[0].reference_center_eV + 0.45
    )
    assert shifted.spectrum_fits[1].energy_shift_eV == result.spectrum_fits[1].energy_shift_eV
    assert shifted.spectrum_fits[1].global_fit.component_areas == result.spectrum_fits[1].global_fit.component_areas


def test_global_series_project_round_trip(tmp_path) -> None:
    spectra, backgrounds, components = _series()
    result = fit_global_series(
        spectra[:2],
        backgrounds[:2],
        components,
        workspace_indices=(0, 1),
        settings=GlobalSeriesSettings(max_nfev=1_000),
    )
    items = tuple(
        WorkspaceItemData(
            spectrum=spectrum,
            background_method=BackgroundMethod.LINEAR,
            background_parameters={"edge_fraction": 0.03, "edge_statistic": "lower_quantile"},
        )
        for spectrum in spectra[:2]
    )
    destination = tmp_path / "global-series.xpsa"
    save_project(
        destination,
        spectrum=spectra[0],
        background_method=BackgroundMethod.LINEAR,
        background_parameters=items[0].background_parameters,
        peaks=(),
        workspace_items=items,
        active_workspace_index=0,
        global_series_result=result,
    )

    loaded = load_project(destination)
    assert loaded.global_series_result is not None
    restored = loaded.global_series_result
    assert restored.workspace_indices == (0, 1)
    assert restored.settings.maximum_energy_shift_eV == pytest.approx(0.25)
    assert restored.component_trends[0].areas == pytest.approx(result.component_trends[0].areas)
    assert np.allclose(
        restored.spectrum_fits[1].global_fit.fitted_intensity,
        result.spectrum_fits[1].global_fit.fitted_intensity,
    )


def test_global_series_accepts_different_sampling_grids_for_the_same_region() -> None:
    energy_a = np.linspace(280.0, 290.0, 301)
    energy_b = np.linspace(279.5, 290.5, 443)
    background_a = np.full(energy_a.size, 2.0)
    background_b = np.full(energy_b.size, 2.5)
    first = Spectrum(
        energy_a,
        background_a + evaluate_peak(energy_a, profile="gaussian", center=285.0, fwhm=1.25, area=240.0),
        label="Grid A",
    )
    second = Spectrum(
        energy_b,
        background_b + evaluate_peak(energy_b, profile="gaussian", center=285.07, fwhm=1.25, area=410.0),
        label="Grid B",
    )
    component = PeakComponent(
        "C 1s",
        center=285.0,
        fwhm=1.3,
        area=250.0,
        profile="gaussian",
        center_bounds=(283.0, 287.0),
        fwhm_bounds=(0.5, 2.5),
        area_bounds=(1.0, 1_000.0),
    )
    result = fit_global_series(
        (first, second),
        (background_a, background_b),
        (component,),
        settings=GlobalSeriesSettings(max_nfev=600),
    )

    assert result.total_samples == 744
    assert result.spectrum_fits[0].global_fit.fitted_intensity.size == 301
    assert result.spectrum_fits[1].global_fit.fitted_intensity.size == 443
    assert result.spectrum_fits[1].energy_shift_eV == pytest.approx(0.07, abs=0.005)
