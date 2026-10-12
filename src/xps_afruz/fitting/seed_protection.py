"""Guardrails for analyst-seeded, multi-component XPS fits.

Ordinary nonlinear least squares treats same-profile peaks as exchangeable.  If
every center and width is given the full region as its bounds, components can
cross, collect in one residual minimum, or let one very broad peak absorb the
envelope.  The helpers here preserve the analyst's starting topology without
fixing the fitted values: each seed receives an ordered local center zone and a
finite, locally scaled width range.

This is a numerical identifiability aid, not evidence that a component is a
real chemical state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import FitResult, ParameterTie, PeakComponent


@dataclass(frozen=True, slots=True)
class SeedProtectionDiagnostics:
    """Transparent warnings for a protected manual fit."""

    collapsed_pairs: tuple[tuple[str, str], ...]
    weak_components: tuple[str, ...]
    center_boundary_components: tuple[str, ...]
    width_boundary_components: tuple[str, ...]
    residual_autocorrelation: float

    @property
    def has_pile_up(self) -> bool:
        return bool(self.collapsed_pairs)

    @property
    def review_needed(self) -> bool:
        return bool(
            self.collapsed_pairs
            or self.weak_components
            or self.center_boundary_components
            or self.width_boundary_components
            or abs(self.residual_autocorrelation) >= 0.50
        )


def protect_manual_peak_seeds(
    energy: ArrayLike,
    components: Sequence[PeakComponent],
    ties: Iterable[ParameterTie] = (),
    *,
    seed_centers: Mapping[str, float] | None = None,
) -> tuple[PeakComponent, ...]:
    """Return components with ordered local bounds around their current seeds.

    Adjacent seeds define Voronoi-like center zones separated by a small dead
    band.  Centers therefore remain adjustable but cannot exchange identities
    or converge to the same coordinate.  Width ceilings scale with the current
    width and nearest-neighbour spacing instead of the entire ROI.

    Existing user bounds are always respected.  Center ties are propagated
    back to their source when possible so a constrained doublet remains inside
    the protected zones of both members.
    """

    axis = _validated_energy(energy)
    if not components:
        raise ValueError("Seed protection requires at least one peak component.")
    names = [component.name for component in components]
    if len(names) != len(set(names)):
        raise ValueError("Seed protection requires uniquely named components.")

    low = float(axis[0])
    high = float(axis[-1])
    span = high - low
    step = max(abs(float(np.median(np.diff(axis)))), np.finfo(float).eps)
    anchors = {
        component.name: float(
            component.center if seed_centers is None else seed_centers.get(component.name, component.center)
        )
        for component in components
    }
    if not all(low <= anchor <= high and np.isfinite(anchor) for anchor in anchors.values()):
        raise ValueError("Every protected peak seed must be finite and inside the energy region.")
    ordered = sorted(enumerate(components), key=lambda item: (anchors[item[1].name], item[0]))
    centers = np.asarray([anchors[component.name] for _, component in ordered], dtype=float)

    zone_lowers = np.full(len(components), low, dtype=float)
    zone_uppers = np.full(len(components), high, dtype=float)
    for position in range(len(ordered) - 1):
        left_center = centers[position]
        right_center = centers[position + 1]
        distance = right_center - left_center
        if distance <= 0.0:
            # Identical seeds carry no spatial information.  Give each a tiny,
            # deterministic side of the shared coordinate instead of creating
            # invalid zero-width bounds.  The diagnostics will still identify
            # the resulting pair as unresolved.
            distance = step
            midpoint = left_center
            gap = min(step * 0.25, span * 1e-4)
        else:
            midpoint = (left_center + right_center) / 2.0
            gap = min(max(step * 1.5, distance * 0.04), distance * 0.20)
        zone_uppers[position] = min(zone_uppers[position], midpoint - gap / 2.0)
        zone_lowers[position + 1] = max(zone_lowers[position + 1], midpoint + gap / 2.0)

    preliminary: dict[str, tuple[float, float]] = {}
    for position, (_, component) in enumerate(ordered):
        if len(components) == 1:
            radius = max(component.fwhm * 3.0, span * 0.25, step * 6.0)
        else:
            neighbour_distances: list[float] = []
            if position:
                neighbour_distances.append(max(centers[position] - centers[position - 1], step))
            if position + 1 < len(components):
                neighbour_distances.append(max(centers[position + 1] - centers[position], step))
            local_spacing = min(neighbour_distances)
            radius = max(component.fwhm * 2.5, local_spacing * 0.48, step * 6.0)
        anchor = anchors[component.name]
        lower = max(low, float(component.center_bounds[0]), float(zone_lowers[position]), anchor - radius)
        upper = min(high, float(component.center_bounds[1]), float(zone_uppers[position]), anchor + radius)
        preliminary[component.name] = _bounds_containing_value(lower, upper, component.center, low, high, step)

    center_bounds = dict(preliminary)
    center_ties = [
        tie
        for tie in ties
        if tie.target.endswith(".center") and tie.source.endswith(".center") and abs(float(tie.scale)) > 1e-15
    ]
    for _ in range(max(1, len(center_ties))):
        changed = False
        for tie in center_ties:
            target_name = tie.target.removesuffix(".center")
            source_name = tie.source.removesuffix(".center")
            if target_name not in center_bounds or source_name not in center_bounds:
                continue
            target_lower, target_upper = center_bounds[target_name]
            translated = sorted(
                (
                    (target_lower - float(tie.offset)) / float(tie.scale),
                    (target_upper - float(tie.offset)) / float(tie.scale),
                )
            )
            source_lower, source_upper = center_bounds[source_name]
            candidate = (max(source_lower, translated[0]), min(source_upper, translated[1]))
            source_component = next(component for component in components if component.name == source_name)
            if candidate[0] < candidate[1] and candidate[0] <= source_component.center <= candidate[1]:
                if candidate != center_bounds[source_name]:
                    center_bounds[source_name] = candidate
                    changed = True
        if not changed:
            break

    protected: list[PeakComponent | None] = [None] * len(components)
    for position, (original_index, component) in enumerate(ordered):
        distances: list[float] = []
        if position:
            distances.append(max(centers[position] - centers[position - 1], step))
        if position + 1 < len(components):
            distances.append(max(centers[position + 1] - centers[position], step))
        local_spacing = min(distances) if distances else span
        lower_width = max(
            float(component.fwhm_bounds[0]),
            min(component.fwhm * 0.80, max(step * 1.5, 0.01)),
        )
        proposed_upper = max(component.fwhm * 3.25, local_spacing * 0.80, step * 8.0)
        upper_width = min(float(component.fwhm_bounds[1]), span * 0.50, proposed_upper)
        upper_width = max(upper_width, component.fwhm * 1.05, lower_width * 1.01)
        if np.isfinite(component.fwhm_bounds[1]):
            upper_width = min(upper_width, float(component.fwhm_bounds[1]))
        if upper_width <= lower_width:
            upper_width = max(lower_width * 1.01, component.fwhm * 1.001)

        protected[original_index] = replace(
            component,
            center_bounds=center_bounds[component.name],
            fwhm_bounds=(lower_width, upper_width),
        )
    return tuple(component for component in protected if component is not None)


def diagnose_seeded_fit(
    result: FitResult,
    components: Sequence[PeakComponent],
    energy: ArrayLike,
) -> SeedProtectionDiagnostics:
    """Report unresolved overlap, weak peaks, and guardrail boundary hits."""

    axis = _validated_energy(energy)
    step = max(abs(float(np.median(np.diff(axis)))), np.finfo(float).eps)
    by_name = {component.name: component for component in components}
    fitted = sorted(
        [
            (
            name,
            float(result.parameter_values[f"{name}.center"]),
            float(result.parameter_values[f"{name}.fwhm"]),
            float(result.parameter_values[f"{name}.area"]),
            )
            for name in by_name
        ],
        key=lambda item: item[1],
    )
    collapsed: list[tuple[str, str]] = []
    for left, right in zip(fitted, fitted[1:], strict=False):
        mean_width = max((left[2] + right[2]) / 2.0, np.finfo(float).eps)
        if abs(right[1] - left[1]) < max(step * 2.0, mean_width * 0.22):
            collapsed.append((left[0], right[0]))

    total_area = sum(max(entry[3], 0.0) for entry in fitted)
    weak = tuple(entry[0] for entry in fitted if total_area > 0.0 and entry[3] / total_area < 0.002)
    center_hits: list[str] = []
    width_hits: list[str] = []
    for name, center, width, _area in fitted:
        component = by_name[name]
        center_tolerance = max(step * 1.5, (component.center_bounds[1] - component.center_bounds[0]) * 0.003)
        if center - component.center_bounds[0] <= center_tolerance or component.center_bounds[1] - center <= center_tolerance:
            center_hits.append(name)
        width_tolerance = max(step * 0.5, (component.fwhm_bounds[1] - component.fwhm_bounds[0]) * 0.003)
        if component.fwhm_bounds[1] - width <= width_tolerance:
            width_hits.append(name)

    residual = np.asarray(result.residual, dtype=float)
    if residual.size >= 4 and np.std(residual[:-1]) > 0.0 and np.std(residual[1:]) > 0.0:
        autocorrelation = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    else:
        autocorrelation = 0.0
    return SeedProtectionDiagnostics(
        collapsed_pairs=tuple(collapsed),
        weak_components=weak,
        center_boundary_components=tuple(center_hits),
        width_boundary_components=tuple(width_hits),
        residual_autocorrelation=autocorrelation,
    )


def _validated_energy(energy: ArrayLike) -> NDArray[np.float64]:
    axis = np.asarray(energy, dtype=float).reshape(-1)
    if axis.size < 4 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise ValueError("Seed protection requires a finite, strictly increasing energy axis.")
    return axis


def _bounds_containing_value(
    lower: float,
    upper: float,
    value: float,
    spectrum_low: float,
    spectrum_high: float,
    step: float,
) -> tuple[float, float]:
    """Repair only pathological duplicate/edge seed zones deterministically."""

    if lower < upper and lower <= value <= upper:
        return float(lower), float(upper)
    padding = max(step * 0.25, (spectrum_high - spectrum_low) * 1e-8)
    repaired_lower = max(spectrum_low, min(lower, value - padding))
    repaired_upper = min(spectrum_high, max(upper, value + padding))
    if repaired_lower >= repaired_upper:
        repaired_lower = value - padding
        repaired_upper = value + padding
    return float(repaired_lower), float(repaired_upper)
