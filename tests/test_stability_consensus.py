import numpy as np
import pytest

from xps_afruz import PeakProfile, Spectrum
from xps_afruz.fitting.peak_shapes import evaluate_peak
from xps_afruz.fitting.stability_consensus import (
    StabilityConsensusCancelled,
    StabilityConsensusSettings,
    _Observation,
    _RunRecord,
    _component_metrics,
    stability_consensus,
)


def _separated_doublet() -> tuple[Spectrum, np.ndarray]:
    energy = np.linspace(280.0, 290.0, 240)
    background = 3.0 + 0.02 * (energy - energy[0])
    observed = background.copy()
    observed += evaluate_peak(
        energy,
        profile=PeakProfile.GAUSSIAN,
        center=283.6,
        fwhm=0.90,
        area=120.0,
    )
    observed += evaluate_peak(
        energy,
        profile=PeakProfile.GAUSSIAN,
        center=286.4,
        fwhm=1.10,
        area=180.0,
    )
    return Spectrum(energy, observed, energy_uncertainty_eV=0.04), background


def test_consensus_reports_stable_components_and_peak_count_support() -> None:
    spectrum, background = _separated_doublet()
    result = stability_consensus(
        spectrum,
        background,
        StabilityConsensusSettings(
            peak_count=2,
            profile=PeakProfile.GAUSSIAN,
            ensemble_runs=9,
            candidate_trials_per_run=4,
            max_nfev=400,
            compare_neighbor_counts=True,
            bootstrap_strength=0.0,
            background_modes=("current", "linear"),
            profiles=(PeakProfile.GAUSSIAN,),
            random_seed=12,
        ),
    )

    assert result.successful_runs >= 8
    assert result.paired_model_scenarios >= 4
    assert result.preferred_peak_count == 2
    assert result.peak_count_probabilities[2] > 0.95
    assert sum(result.bic_peak_count_probabilities.values()) == pytest.approx(1.0)
    assert sum(result.aicc_peak_count_probabilities.values()) == pytest.approx(1.0)
    assert result.peak_count_probabilities[2] == pytest.approx(
        0.5
        * (
            result.bic_peak_count_probabilities[2]
            + result.aicc_peak_count_probabilities[2]
        )
    )
    assert len(result.components) == 2
    assert [component.center_mean_eV for component in result.components] == pytest.approx(
        [283.6, 286.4], abs=0.12
    )
    assert all(component.survival_probability >= 0.80 for component in result.components)
    assert all(component.trust_score >= 75.0 for component in result.components)
    assert all(component.identifiability != "UNSUPPORTED / UNSTABLE" for component in result.components)
    assert "do not prove" in result.numerical_note


def test_consensus_can_be_cancelled_before_mutating_any_fit() -> None:
    spectrum, background = _separated_doublet()
    with pytest.raises(StabilityConsensusCancelled):
        stability_consensus(
            spectrum,
            background,
            StabilityConsensusSettings(
                peak_count=2,
                ensemble_runs=6,
                candidate_trials_per_run=4,
                compare_neighbor_counts=False,
                background_modes=("current",),
                profiles=(PeakProfile.GAUSSIAN,),
            ),
            cancellation_callback=lambda: True,
        )


def test_component_collapse_is_counted_as_risk_not_survival() -> None:
    stable = _Observation(center=285.0, fwhm=1.0, area=100.0, boundary_hit=False, collapsed=False)
    collapsed = _Observation(center=285.0, fwhm=0.01, area=0.001, boundary_hit=True, collapsed=True)
    records = (
        _RunRecord(2, "Linear", "gaussian", 0.0, (stable,), 10.0, 10.0, 200),
        _RunRecord(2, "Shirley", "gaussian", 0.0, (collapsed,), 11.0, 11.0, 200),
    )

    component = _component_metrics(
        name="Consensus Peak 1",
        samples=((records[0], stable), (records[1], collapsed)),
        requested_peak_count=2,
        requested_successful_runs=2,
        counts={2: 1},
        total_by_count={2: 2},
        peak_count_probabilities={2: 1.0},
        step=0.05,
    )

    assert component.survival_probability == pytest.approx(0.5)
    assert component.collapse_frequency == pytest.approx(0.5)
    assert component.sample_count == 2
