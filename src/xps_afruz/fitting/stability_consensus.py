"""Perturbation-ensemble stability analysis for XPS peak models.

The consensus engine does not establish that a component is a chemical state.
It asks a narrower, auditable numerical question: does a comparable component
survive reasonable changes in noise, calibration, fit window, background,
starting position, line shape, and (optionally) neighbouring peak counts?
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ..backgrounds import BackgroundMethod, EdgeStatistic, TougaardVariant, diagnose_background, estimate_background
from ..core.spectrum import Spectrum
from .peak_shapes import PeakProfile
from .statistics import fit_information_criteria
from .stability_models import (
    ComponentStability,
    StabilityConsensusCancelled,
    StabilityConsensusResult,
    StabilityConsensusSettings,
    _Observation,
    _RunRecord,
)
from .smart_deconvolution import (
    SmartDeconvolutionCancelled,
    SmartDeconvolutionSettings,
    smart_deconvolve,
)

ProgressCallback = Callable[[float, int, str], None]
CancellationCallback = Callable[[], bool]

def stability_consensus(
    spectrum: Spectrum,
    current_background: NDArray[np.float64],
    settings: StabilityConsensusSettings,
    *,
    current_background_label: str = "Current",
    progress_callback: ProgressCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> StabilityConsensusResult:
    """Run a deterministic ensemble and aggregate component stability.

    BIC/AICc support is an approximate numerical model comparison across
    perturbed replicas. It is not a posterior probability of chemistry or a
    substitute for physically justified constraints.
    """

    retained_background = np.asarray(current_background, dtype=float).reshape(-1)
    if retained_background.size != spectrum.size or not np.all(np.isfinite(retained_background)):
        raise ValueError("Consensus analysis needs a finite current background aligned with the spectrum.")
    if settings.weighting == "poisson" and np.any(spectrum.intensity < 0.0):
        raise ValueError("Poisson consensus analysis requires non-negative raw-count data.")

    rng = np.random.default_rng(settings.random_seed)
    counts = settings.peak_counts
    profiles = settings.profiles_to_test
    background_schedule = _balanced_schedule(settings.background_modes, settings.ensemble_runs, rng)
    profile_schedule = _balanced_schedule(profiles, settings.ensemble_runs, rng)
    background_schedule[0] = "current"
    profile_schedule[0] = settings.profile

    records: list[_RunRecord] = []
    failures = 0
    completed_model_runs = 0
    total_model_runs = settings.ensemble_runs * len(counts)
    for scenario_index in range(settings.ensemble_runs):
        if cancellation_callback is not None and cancellation_callback():
            raise StabilityConsensusCancelled("Stability-consensus analysis was cancelled.")
        background_mode = str(background_schedule[scenario_index])
        profile = PeakProfile(profile_schedule[scenario_index])
        run_rng = np.random.default_rng(settings.random_seed + 104_729 * (scenario_index + 1))
        baseline = scenario_index == 0
        try:
            simulated, selection, calibration_shift = _perturbed_spectrum(
                spectrum,
                settings,
                run_rng,
                baseline=baseline,
            )
            background, background_label = _scenario_background(
                simulated,
                retained_background[selection],
                background_mode,
                current_background_label=current_background_label,
            )
            diagnostics = diagnose_background(simulated, background)
            if background_mode != "current" and diagnostics.fraction_above_signal > settings.max_background_above_fraction:
                raise ValueError("Perturbed background exceeded the accepted diagnostic threshold.")
        except (SmartDeconvolutionCancelled, StabilityConsensusCancelled):
            raise StabilityConsensusCancelled("Stability-consensus analysis was cancelled.")
        except (FloatingPointError, ValueError):
            failures += len(counts)
            completed_model_runs += len(counts)
            if progress_callback is not None:
                progress_callback(
                    completed_model_runs,
                    total_model_runs,
                    f"Scenario {scenario_index + 1}/{settings.ensemble_runs} rejected by background diagnostics",
                )
            continue

        for peak_count in counts:
            if cancellation_callback is not None and cancellation_callback():
                raise StabilityConsensusCancelled("Stability-consensus analysis was cancelled.")
            smart_settings = SmartDeconvolutionSettings(
                peak_count=peak_count,
                profile=profile,
                candidate_trials=settings.candidate_trials_per_run,
                max_nfev=settings.max_nfev,
                weighting=settings.weighting,
                count_scale=settings.count_scale,
                random_seed=settings.random_seed + 7_919 * (scenario_index + 1) + peak_count,
            )

            def report_inner(completed: int, total: int, message: str) -> None:
                if progress_callback is None:
                    return
                fractional = min(max(completed / max(total, 1), 0.0), 1.0)
                progress_callback(
                    completed_model_runs + fractional,
                    total_model_runs,
                    f"Scenario {scenario_index + 1}/{settings.ensemble_runs} · K={peak_count} · {background_label} · {profile.display_name}",
                )

            try:
                smart = smart_deconvolve(
                    simulated,
                    background,
                    smart_settings,
                    progress_callback=report_inner,
                    cancellation_callback=cancellation_callback,
                )
                if not smart.fit_result.success:
                    raise ValueError("Candidate optimizer did not converge.")
                observations = _observations_from_fit(simulated, smart.fit_result.parameter_values, smart.components)
                records.append(
                    _RunRecord(
                        peak_count=peak_count,
                        background=background_label,
                        profile=profile.value,
                        calibration_shift_eV=calibration_shift,
                        observations=observations,
                        bic=float(smart.bic),
                        aicc=fit_information_criteria(smart.fit_result, simulated.size).aicc,
                        sample_count=simulated.size,
                        scenario_index=scenario_index,
                    )
                )
            except SmartDeconvolutionCancelled:
                raise StabilityConsensusCancelled("Stability-consensus analysis was cancelled.")
            except (FloatingPointError, ValueError):
                failures += 1
            completed_model_runs += 1
            if progress_callback is not None:
                progress_callback(
                    completed_model_runs,
                    total_model_runs,
                    f"Completed {completed_model_runs}/{total_model_runs} model fits · {len(records)} valid",
                )

    minimum = max(3, int(np.ceil(total_model_runs * settings.minimum_success_fraction)))
    requested_records = [record for record in records if record.peak_count == settings.peak_count]
    minimum_requested = max(1, int(np.ceil(settings.ensemble_runs * settings.minimum_success_fraction)))
    if len(records) < minimum or len(requested_records) < minimum_requested:
        raise ValueError(
            f"Only {len(records)}/{total_model_runs} consensus model fits were valid; at least {minimum} total and "
            f"{minimum_requested} requested-K fits are required. Review the ROI/background or reduce model complexity."
        )

    (
        probabilities,
        bic_probabilities,
        aicc_probabilities,
        bic_by_count,
        aicc_by_count,
        paired_scenarios,
    ) = _peak_count_support(records, counts)
    minimum_paired = max(3, int(np.ceil(settings.ensemble_runs * settings.minimum_success_fraction)))
    if paired_scenarios < minimum_paired:
        raise ValueError(
            f"Only {paired_scenarios}/{settings.ensemble_runs} scenarios completed every tested peak count; "
            f"at least {minimum_paired} complete paired scenarios are required for BIC/AICc support."
        )
    components = _aggregate_components(spectrum, settings, records, requested_records, probabilities)
    return StabilityConsensusResult(
        spectrum_label=spectrum.label,
        requested_peak_count=settings.peak_count,
        peak_count_probabilities=probabilities,
        bic_peak_count_probabilities=bic_probabilities,
        aicc_peak_count_probabilities=aicc_probabilities,
        bic_by_peak_count=bic_by_count,
        aicc_by_peak_count=aicc_by_count,
        components=components,
        requested_runs=settings.ensemble_runs,
        successful_runs=len(records),
        failed_runs=failures,
        paired_model_scenarios=paired_scenarios,
        candidate_trials_per_run=settings.candidate_trials_per_run,
        random_seed=settings.random_seed,
        background_models=tuple(_background_display_name(mode, current_background_label) for mode in settings.background_modes),
        profiles=tuple(profile.value for profile in profiles),
        weighting=settings.weighting,
        count_scale=float(settings.count_scale) if settings.weighting == "poisson" else None,
        calibration_uncertainty_eV=spectrum.energy_uncertainty_eV,
        roi_jitter_fraction=settings.roi_jitter_fraction,
        bootstrap_strength=settings.bootstrap_strength,
        calibration_sigma_scale=settings.calibration_sigma_scale,
        max_nfev=settings.max_nfev,
        max_background_above_fraction=settings.max_background_above_fraction,
        numerical_note=(
            "Trust scores summarize perturbation stability and approximate BIC/AICc support. They do not prove peak "
            "existence, elemental identity, oxidation state, or a unique physical interpretation."
        ),
    )


def _balanced_schedule(values: Sequence[object], count: int, rng: np.random.Generator) -> list[object]:
    output: list[object] = []
    while len(output) < count:
        block = list(values)
        rng.shuffle(block)
        output.extend(block)
    return output[:count]


def _perturbed_spectrum(
    spectrum: Spectrum,
    settings: StabilityConsensusSettings,
    rng: np.random.Generator,
    *,
    baseline: bool,
) -> tuple[Spectrum, slice, float]:
    sample_count = spectrum.size
    maximum_crop = int(np.floor(sample_count * settings.roi_jitter_fraction))
    if baseline or maximum_crop < 1:
        left_crop = right_crop = 0
    else:
        left_crop = int(rng.integers(0, maximum_crop + 1))
        right_crop = int(rng.integers(0, maximum_crop + 1))
    minimum_points = max(40, settings.peak_count * 12)
    if sample_count - left_crop - right_crop < minimum_points:
        left_crop = right_crop = 0
    selection = slice(left_crop, sample_count - right_crop if right_crop else None)

    if baseline or spectrum.energy_uncertainty_eV <= 0.0:
        calibration_shift = 0.0
    else:
        calibration_shift = float(
            rng.normal(0.0, spectrum.energy_uncertainty_eV * settings.calibration_sigma_scale)
        )
    intensity = spectrum.intensity[selection].copy()
    if not baseline and settings.bootstrap_strength > 0.0:
        intensity = _bootstrap_intensity(
            intensity,
            weighting=settings.weighting,
            count_scale=settings.count_scale,
            strength=settings.bootstrap_strength,
            rng=rng,
        )
    return (
        Spectrum(
            spectrum.energy[selection] + calibration_shift,
            intensity,
            label=f"{spectrum.label} · stability replica",
            energy_unit=spectrum.energy_unit,
            intensity_unit=spectrum.intensity_unit,
            metadata=spectrum.metadata,
            energy_scale=spectrum.energy_scale,
            photon_energy_eV=spectrum.photon_energy_eV,
            energy_uncertainty_eV=spectrum.energy_uncertainty_eV,
            original_energy=spectrum.original_energy[selection],
            original_energy_scale=spectrum.original_energy_scale,
            correction_history=spectrum.correction_history,
            processing_history=spectrum.processing_history,
        ),
        selection,
        calibration_shift,
    )


def _bootstrap_intensity(
    intensity: NDArray[np.float64],
    *,
    weighting: str,
    count_scale: float,
    strength: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if weighting == "poisson":
        physical = np.clip(intensity * count_scale, 0.0, None)
        sampled = rng.poisson(physical).astype(float) / count_scale
        return intensity + strength * (sampled - intensity)
    differences = np.diff(intensity)
    median = float(np.median(differences))
    centered = (differences - median) / np.sqrt(2.0)
    sigma = 1.4826 * float(np.median(np.abs(differences - median))) / np.sqrt(2.0)
    if sigma <= np.finfo(float).eps:
        return intensity.copy()
    # The robust clip prevents genuine peak slopes from being resampled as
    # artificial narrow peaks while retaining the empirical high-frequency
    # noise distribution rather than assuming it is Gaussian.
    residual_pool = np.clip(centered, -4.0 * sigma, 4.0 * sigma)
    residual_pool = residual_pool - np.median(residual_pool)
    return intensity + strength * rng.choice(residual_pool, size=intensity.size, replace=True)


def _scenario_background(
    spectrum: Spectrum,
    retained_background: NDArray[np.float64],
    mode: str,
    *,
    current_background_label: str,
) -> tuple[NDArray[np.float64], str]:
    if mode == "current":
        return retained_background.copy(), _background_display_name(mode, current_background_label)
    common = {"edge_fraction": 0.03, "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value}
    if mode == "linear":
        values = estimate_background(spectrum, BackgroundMethod.LINEAR, **common)
    elif mode == "shirley":
        values = estimate_background(
            spectrum,
            BackgroundMethod.SHIRLEY,
            **common,
            max_iterations=200,
            tolerance=1e-7,
        )
    elif mode == "tougaard_u2":
        values = estimate_background(
            spectrum,
            BackgroundMethod.TOUGAARD,
            **common,
            variant=TougaardVariant.U2.value,
            b=2866.0,
            c=1643.0,
            d=0.0,
            scale=None,
        )
    elif mode == "tougaard_u3":
        values = estimate_background(
            spectrum,
            BackgroundMethod.TOUGAARD,
            **common,
            variant=TougaardVariant.U3.value,
            b=2866.0,
            c=1643.0,
            d=250.0,
            scale=None,
        )
    else:
        raise ValueError(f"Unknown consensus background mode: {mode}")
    return values, _background_display_name(mode, current_background_label)


def _background_display_name(mode: str, current_background_label: str) -> str:
    labels = {
        "current": f"Current: {current_background_label}",
        "linear": "Linear",
        "shirley": "Shirley",
        "tougaard_u2": "Tougaard U2",
        "tougaard_u3": "Tougaard U3 (D=250)",
    }
    return labels[mode]


def _observations_from_fit(
    spectrum: Spectrum,
    values: Mapping[str, float],
    components: Sequence[object],
) -> tuple[_Observation, ...]:
    step = max(float(np.median(np.diff(spectrum.energy))), 1e-12)
    span = float(spectrum.energy[-1] - spectrum.energy[0])
    total_area = sum(max(float(value), 0.0) for name, value in values.items() if name.endswith(".area"))
    observations: list[_Observation] = []
    for component in components:
        name = str(getattr(component, "name"))
        center = float(values[f"{name}.center"])
        fwhm = float(values[f"{name}.fwhm"])
        area = float(values[f"{name}.area"])
        center_bounds = tuple(getattr(component, "center_bounds"))
        fwhm_bounds = tuple(getattr(component, "fwhm_bounds"))
        area_bounds = tuple(getattr(component, "area_bounds"))
        boundary_hit = (
            _near_bound(center, center_bounds)
            or _near_bound(fwhm, fwhm_bounds)
            or _near_bound(area, area_bounds)
            or center <= spectrum.energy[0] + max(2.0 * step, span * 0.005)
            or center >= spectrum.energy[-1] - max(2.0 * step, span * 0.005)
        )
        collapsed = fwhm <= step * 1.55 or area <= max(total_area * 1e-4, np.finfo(float).eps)
        observations.append(_Observation(center, fwhm, area, boundary_hit, collapsed))
    observations.sort(key=lambda observation: observation.center)
    mutable = list(observations)
    for index in range(len(mutable) - 1):
        left, right = mutable[index], mutable[index + 1]
        separation = (right.center - left.center) / max((left.fwhm + right.fwhm) / 2.0, 1e-12)
        if separation < 0.28:
            mutable[index] = _Observation(left.center, left.fwhm, left.area, left.boundary_hit, True)
            mutable[index + 1] = _Observation(right.center, right.fwhm, right.area, right.boundary_hit, True)
    return tuple(mutable)


def _near_bound(value: float, bounds: tuple[float, float]) -> bool:
    lower, upper = float(bounds[0]), float(bounds[1])
    span = upper - lower
    if not np.isfinite(span) or span <= 0.0:
        return False
    return min(value - lower, upper - value) <= span * 0.005


def _peak_count_support(
    records: Sequence[_RunRecord],
    expected_counts: Sequence[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float], dict[int, float], int]:
    expected = set(expected_counts)
    by_scenario: dict[int, list[_RunRecord]] = {}
    for record in records:
        by_scenario.setdefault(record.scenario_index, []).append(record)
    complete_scenarios = [
        scenario_records
        for scenario_records in by_scenario.values()
        if {record.peak_count for record in scenario_records} == expected
    ]
    paired_records = [record for scenario_records in complete_scenarios for record in scenario_records]
    if not paired_records:
        raise ValueError("No perturbation scenario completed every requested peak count.")
    counts = sorted(expected)
    representative_n = float(np.median([record.sample_count for record in paired_records]))
    bic_by_count: dict[int, float] = {}
    aicc_by_count: dict[int, float] = {}
    for count in counts:
        selected = [record for record in paired_records if record.peak_count == count]
        bic_by_count[count] = float(np.median([record.bic / record.sample_count for record in selected]) * representative_n)
        aicc_by_count[count] = float(np.median([record.aicc / record.sample_count for record in selected]) * representative_n)
    bic_probabilities = _criterion_weights(bic_by_count)
    aicc_probabilities = _criterion_weights(aicc_by_count)
    # BIC and AICc encode different complexity penalties. Their mean is kept as
    # an explicitly labelled consensus support weight; the individual weights
    # remain available so an analyst can see criterion disagreement.
    probabilities = {
        count: 0.5 * (bic_probabilities[count] + aicc_probabilities[count])
        for count in counts
    }
    return (
        probabilities,
        bic_probabilities,
        aicc_probabilities,
        bic_by_count,
        aicc_by_count,
        len(complete_scenarios),
    )


def _criterion_weights(criteria: Mapping[int, float]) -> dict[int, float]:
    minimum = min(criteria.values())
    raw = {
        count: float(np.exp(np.clip(-0.5 * (criterion - minimum), -700.0, 0.0)))
        for count, criterion in criteria.items()
    }
    total = sum(raw.values())
    return {count: value / total for count, value in raw.items()}


def _aggregate_components(
    spectrum: Spectrum,
    settings: StabilityConsensusSettings,
    records: Sequence[_RunRecord],
    requested_records: Sequence[_RunRecord],
    peak_count_probabilities: Mapping[int, float],
) -> tuple[ComponentStability, ...]:
    reference_record = min(requested_records, key=lambda record: record.bic / record.sample_count)
    reference = reference_record.observations
    step = max(float(np.median(np.diff(spectrum.energy))), 1e-12)
    span = float(spectrum.energy[-1] - spectrum.energy[0])
    matched: list[list[tuple[_RunRecord, _Observation]]] = [[] for _ in reference]
    matched_by_count: list[dict[int, int]] = [dict() for _ in reference]
    total_by_count = {count: sum(record.peak_count == count for record in records) for count in peak_count_probabilities}

    for record in records:
        if not record.observations:
            continue
        cost = np.empty((len(reference), len(record.observations)), dtype=float)
        for row, anchor in enumerate(reference):
            for column, observation in enumerate(record.observations):
                center_cost = abs(observation.center - anchor.center) / max(anchor.fwhm, step * 4.0)
                width_cost = 0.20 * abs(np.log(max(observation.fwhm, step) / max(anchor.fwhm, step)))
                cost[row, column] = center_cost + width_cost
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns, strict=True):
            anchor = reference[int(row)]
            observation = record.observations[int(column)]
            tolerance = max(anchor.fwhm * 1.50, span / max(settings.peak_count * 5.0, 5.0), step * 5.0)
            if abs(observation.center - anchor.center) > tolerance:
                continue
            matched[int(row)].append((record, observation))
            if not observation.collapsed:
                matched_by_count[int(row)][record.peak_count] = (
                    matched_by_count[int(row)].get(record.peak_count, 0) + 1
                )

    output: list[ComponentStability] = []
    for index, samples in enumerate(matched, start=1):
        output.append(
            _component_metrics(
                name=f"Consensus Peak {index}",
                samples=samples,
                requested_peak_count=settings.peak_count,
                requested_successful_runs=len(requested_records),
                counts=matched_by_count[index - 1],
                total_by_count=total_by_count,
                peak_count_probabilities=peak_count_probabilities,
                step=step,
            )
        )
    return tuple(output)


def _component_metrics(
    *,
    name: str,
    samples: Sequence[tuple[_RunRecord, _Observation]],
    requested_peak_count: int,
    requested_successful_runs: int,
    counts: Mapping[int, int],
    total_by_count: Mapping[int, int],
    peak_count_probabilities: Mapping[int, float],
    step: float,
) -> ComponentStability:
    if not samples:
        return ComponentStability(
            name=name,
            sample_count=0,
            survival_probability=0.0,
            model_support_probability=0.0,
            center_mean_eV=float("nan"),
            center_std_eV=float("nan"),
            fwhm_mean_eV=float("nan"),
            fwhm_std_eV=float("nan"),
            area_mean=float("nan"),
            area_std=float("nan"),
            area_cv=float("inf"),
            background_sensitivity=1.0,
            background_sensitivity_label="HIGH",
            calibration_sensitivity=1.0,
            calibration_sensitivity_label="HIGH",
            calibration_contribution_eV=float("nan"),
            boundary_hit_frequency=1.0,
            collapse_frequency=1.0,
            max_parameter_correlation=1.0,
            trust_score=0.0,
            identifiability="UNSUPPORTED / UNSTABLE",
        )

    all_observations = [sample[1] for sample in samples]
    stable_samples = [sample for sample in samples if not sample[1].collapsed]
    collapse = float(np.mean([observation.collapsed for observation in all_observations]))
    if not stable_samples:
        return ComponentStability(
            name=name,
            sample_count=len(samples),
            survival_probability=0.0,
            model_support_probability=0.0,
            center_mean_eV=float("nan"),
            center_std_eV=float("nan"),
            fwhm_mean_eV=float("nan"),
            fwhm_std_eV=float("nan"),
            area_mean=float("nan"),
            area_std=float("nan"),
            area_cv=float("inf"),
            background_sensitivity=1.0,
            background_sensitivity_label="HIGH",
            calibration_sensitivity=1.0,
            calibration_sensitivity_label="HIGH",
            calibration_contribution_eV=float("nan"),
            boundary_hit_frequency=float(np.mean([observation.boundary_hit for observation in all_observations])),
            collapse_frequency=collapse,
            max_parameter_correlation=1.0,
            trust_score=0.0,
            identifiability="UNSUPPORTED / UNSTABLE",
        )

    records = [sample[0] for sample in stable_samples]
    observations = [sample[1] for sample in stable_samples]
    centers = np.asarray([observation.center for observation in observations], dtype=float)
    widths = np.asarray([observation.fwhm for observation in observations], dtype=float)
    areas = np.asarray([observation.area for observation in observations], dtype=float)
    shifts = np.asarray([record.calibration_shift_eV for record in records], dtype=float)
    survival = counts.get(requested_peak_count, 0) / max(requested_successful_runs, 1)
    model_support = sum(
        peak_count_probabilities[count] * counts.get(count, 0) / max(total_by_count.get(count, 0), 1)
        for count in peak_count_probabilities
    )
    center_mean = float(np.mean(centers))
    center_std = float(np.std(centers, ddof=1)) if centers.size > 1 else 0.0
    width_mean = float(np.mean(widths))
    width_std = float(np.std(widths, ddof=1)) if widths.size > 1 else 0.0
    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas, ddof=1)) if areas.size > 1 else 0.0
    area_cv = area_std / max(abs(area_mean), np.finfo(float).eps)
    background_sensitivity = _background_sensitivity(records, centers, areas, width_mean, area_mean)
    calibration_sensitivity, calibration_contribution = _calibration_sensitivity(shifts, centers, center_std)
    boundary = float(np.mean([observation.boundary_hit for observation in observations]))
    correlation = _maximum_parameter_correlation(centers, widths, areas)

    center_factor = float(np.exp(-center_std / max(width_mean * 0.35, step * 2.0)))
    area_factor = float(np.exp(-area_cv / 0.50))
    correlation_factor = 1.0 - min(max(correlation - 0.65, 0.0) / 0.35, 1.0)
    trust = 100.0 * (
        0.30 * survival
        + 0.15 * model_support
        + 0.12 * center_factor
        + 0.08 * area_factor
        + 0.10 * (1.0 - background_sensitivity)
        + 0.05 * (1.0 - calibration_sensitivity)
        + 0.08 * (1.0 - boundary)
        + 0.07 * (1.0 - collapse)
        + 0.05 * correlation_factor
    )
    trust = float(np.clip(trust, 0.0, 100.0))
    if trust >= 80.0 and survival >= 0.80 and model_support >= 0.65:
        identifiability = "STRONG NUMERICAL SUPPORT"
    elif trust >= 55.0 and survival >= 0.50:
        identifiability = "PROVISIONAL"
    else:
        identifiability = "UNSUPPORTED / UNSTABLE"
    return ComponentStability(
        name=name,
        sample_count=len(samples),
        survival_probability=float(survival),
        model_support_probability=float(model_support),
        center_mean_eV=center_mean,
        center_std_eV=center_std,
        fwhm_mean_eV=width_mean,
        fwhm_std_eV=width_std,
        area_mean=area_mean,
        area_std=area_std,
        area_cv=float(area_cv),
        background_sensitivity=background_sensitivity,
        background_sensitivity_label=_sensitivity_label(background_sensitivity),
        calibration_sensitivity=calibration_sensitivity,
        calibration_sensitivity_label=_sensitivity_label(calibration_sensitivity),
        calibration_contribution_eV=calibration_contribution,
        boundary_hit_frequency=boundary,
        collapse_frequency=collapse,
        max_parameter_correlation=correlation,
        trust_score=trust,
        identifiability=identifiability,
    )


def _background_sensitivity(
    records: Sequence[_RunRecord],
    centers: NDArray[np.float64],
    areas: NDArray[np.float64],
    mean_width: float,
    mean_area: float,
) -> float:
    labels = sorted({record.background for record in records})
    if len(labels) < 2:
        return 0.0
    center_means: list[float] = []
    area_means: list[float] = []
    for label in labels:
        mask = np.asarray([record.background == label for record in records], dtype=bool)
        if np.any(mask):
            center_means.append(float(np.mean(centers[mask])))
            area_means.append(float(np.mean(areas[mask])))
    center_effect = float(np.ptp(center_means)) / max(mean_width, np.finfo(float).eps)
    area_effect = float(np.ptp(area_means)) / max(abs(mean_area), np.finfo(float).eps)
    return float(np.clip(max(center_effect / 0.75, area_effect / 0.75), 0.0, 1.0))


def _calibration_sensitivity(
    shifts: NDArray[np.float64], centers: NDArray[np.float64], center_std: float
) -> tuple[float, float]:
    shift_variance = float(np.var(shifts))
    if shifts.size < 3 or shift_variance <= np.finfo(float).eps:
        return 0.0, 0.0
    slope = float(np.cov(shifts, centers, ddof=1)[0, 1] / np.var(shifts, ddof=1))
    contribution = abs(slope) * float(np.std(shifts, ddof=1))
    fraction = contribution / max(center_std, np.finfo(float).eps)
    return float(np.clip(fraction, 0.0, 1.0)), float(contribution)


def _maximum_parameter_correlation(*arrays: NDArray[np.float64]) -> float:
    if arrays[0].size < 4:
        return 0.0
    usable = [values for values in arrays if float(np.std(values)) > np.finfo(float).eps]
    if len(usable) < 2:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(np.vstack(usable))
    upper = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
    finite = upper[np.isfinite(upper)]
    return float(np.max(finite)) if finite.size else 0.0


def _sensitivity_label(value: float) -> str:
    if value < 0.15:
        return "LOW"
    if value < 0.40:
        return "MODERATE"
    return "HIGH"
