import numpy as np
import pytest

from xps_afruz import FitModel, PeakComponent, Spectrum, fit_spectrum, make_spin_orbit_doublet
from xps_afruz.fitting.peak_shapes import evaluate_peak
from xps_afruz.fitting.seed_protection import diagnose_seeded_fit, protect_manual_peak_seeds


def test_seed_protection_creates_disjoint_ordered_center_zones_and_local_width_limits() -> None:
    energy = np.linspace(700.0, 740.0, 801)
    components = [
        PeakComponent(
            f"Peak {index}",
            center=center,
            fwhm=width,
            area=area,
            center_bounds=(700.0, 740.0),
            fwhm_bounds=(0.01, 80.0),
        )
        for index, (center, width, area) in enumerate(
            ((712.0, 2.0, 3.0), (722.0, 4.0, 8.0), (734.0, 1.2, 1.0), (736.0, 1.0, 0.8)),
            start=1,
        )
    ]

    protected = sorted(protect_manual_peak_seeds(energy, components), key=lambda component: component.center)

    for component in protected:
        assert component.center_bounds[0] <= component.center <= component.center_bounds[1]
        assert component.fwhm_bounds[0] <= component.fwhm <= component.fwhm_bounds[1]
        assert component.fwhm_bounds[1] <= 20.0
    for left, right in zip(protected, protected[1:], strict=False):
        assert left.center_bounds[1] < right.center_bounds[0]


def test_protected_manual_fit_recovers_distributed_components_without_peak_pile_up() -> None:
    energy = np.linspace(700.0, 740.0, 1_200)
    background = 2.0 + 0.02 * (energy - energy[0])
    observed = background.copy()
    truth = ((712.2, 2.0, 5.0), (722.5, 4.2, 9.0), (734.8, 1.4, 2.5))
    for center, fwhm, area in truth:
        observed += evaluate_peak(energy, profile="pseudo_voigt", center=center, fwhm=fwhm, area=area, eta=0.35)
    spectrum = Spectrum(energy, observed)
    seeds = [
        PeakComponent(
            f"Peak {index}",
            center=center,
            fwhm=fwhm,
            area=area,
            eta=0.35,
            center_bounds=(700.0, 740.0),
            fwhm_bounds=(0.01, 80.0),
        )
        for index, (center, fwhm, area) in enumerate(
            ((711.4, 2.5, 4.0), (721.0, 3.0, 7.0), (735.5, 2.0, 2.0)),
            start=1,
        )
    ]
    protected = protect_manual_peak_seeds(energy, seeds)

    result = fit_spectrum(spectrum, FitModel(list(protected), background))
    fitted_centers = sorted(
        value for name, value in result.parameter_values.items() if name.endswith(".center")
    )
    diagnostics = diagnose_seeded_fit(result, protected, energy)

    assert result.success
    assert fitted_centers == pytest.approx([entry[0] for entry in truth], abs=2e-3)
    assert not diagnostics.has_pile_up
    assert result.residual_rms == pytest.approx(0.0, abs=1e-7)


def test_seed_protection_preserves_center_tied_doublet_motion() -> None:
    energy = np.linspace(700.0, 730.0, 900)
    components, ties = make_spin_orbit_doublet(
        "Metal 2p",
        center=710.0,
        fwhm=1.5,
        area=8.0,
        splitting=3.0,
        upper_to_lower_area_ratio=0.5,
        center_bounds=(700.0, 730.0),
        fwhm_bounds=(0.05, 60.0),
    )

    protected = protect_manual_peak_seeds(energy, components, ties)
    model = FitModel(list(protected), np.zeros_like(energy), ties)
    source = next(parameter for parameter in model.free_parameters() if parameter.name == "Metal 2p lower.center")
    resolved = model.resolved_values({"Metal 2p lower.center": source.value})
    upper = next(component for component in protected if component.name == "Metal 2p upper")

    assert source.lower <= source.value <= source.upper
    assert upper.center_bounds[0] <= resolved["Metal 2p upper.center"] <= upper.center_bounds[1]
    assert protected[0].fwhm_bounds[1] < 60.0


def test_seed_anchors_remain_fixed_when_current_fit_values_move() -> None:
    energy = np.linspace(700.0, 740.0, 801)
    initial = [
        PeakComponent("Left", center=712.0, fwhm=2.0, area=3.0, center_bounds=(700.0, 740.0)),
        PeakComponent("Right", center=722.0, fwhm=3.0, area=5.0, center_bounds=(700.0, 740.0)),
    ]
    first = protect_manual_peak_seeds(energy, initial)
    moved = [
        PeakComponent(
            component.name,
            center=component.center_bounds[1] - 0.01,
            fwhm=component.fwhm,
            area=component.area,
            center_bounds=(700.0, 740.0),
        )
        for component in first
    ]

    second = protect_manual_peak_seeds(
        energy,
        moved,
        seed_centers={"Left": 712.0, "Right": 722.0},
    )

    assert second[0].center_bounds == pytest.approx(first[0].center_bounds)
    assert second[1].center_bounds == pytest.approx(first[1].center_bounds)
