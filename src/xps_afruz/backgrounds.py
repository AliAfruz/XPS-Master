"""Background-estimation routines for XPS regions.

Background choice is a scientific modelling decision. These routines expose
their parameters explicitly and never conceal the chosen method in a fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .core.spectrum import Spectrum


class BackgroundMethod(str, Enum):
    """Supported fixed background estimators."""

    LINEAR = "linear"
    SHIRLEY = "shirley"
    TOUGAARD = "tougaard"
    CUSTOM = "custom"


class TougaardVariant(str, Enum):
    """Explicit Tougaard kernel parameterizations for exploratory fitting."""

    U2 = "u2"
    U3 = "u3"


class EdgeStatistic(str, Enum):
    """How automatic endpoint anchors avoid strong features at region edges."""

    MEDIAN = "median"
    LOWER_QUANTILE = "lower_quantile"


@dataclass(frozen=True, slots=True)
class BackgroundSettings:
    """Serializable background choice and its exact user-visible parameters."""

    method: BackgroundMethod | str = BackgroundMethod.LINEAR
    parameters: Mapping[str, float | int | str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", BackgroundMethod(self.method))
        object.__setattr__(self, "parameters", dict(self.parameters))

    def as_dict(self) -> dict[str, object]:
        return {"method": self.method.value, "parameters": dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class BackgroundDiagnostics:
    """Simple quality checks to make a nonphysical background visible."""

    left_anchor: float
    right_anchor: float
    fraction_above_signal: float
    minimum_net_intensity: float
    robust_net_noise: float

    @property
    def is_cautionary(self) -> bool:
        return self.fraction_above_signal > 0.05


def _arrays(energy: ArrayLike, intensity: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(energy, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if x.size < 4 or x.size != y.size:
        raise ValueError("Background estimation requires equally sized arrays with at least four points.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Energy must be strictly ascending for background estimation.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Background estimation requires finite values.")
    return x, y


def _edge_level(
    values: NDArray[np.float64],
    *,
    left: bool,
    fraction: float,
    statistic: EdgeStatistic | str = EdgeStatistic.MEDIAN,
) -> float:
    count = max(1, int(np.ceil(values.size * fraction)))
    samples = values[:count] if left else values[-count:]
    if EdgeStatistic(statistic) is EdgeStatistic.LOWER_QUANTILE:
        return float(np.quantile(samples, 0.20))
    return float(np.median(samples))


def linear_background(
    energy: ArrayLike,
    intensity: ArrayLike,
    *,
    left_level: float | None = None,
    right_level: float | None = None,
    edge_fraction: float = 0.03,
    edge_statistic: EdgeStatistic | str = EdgeStatistic.MEDIAN,
) -> NDArray[np.float64]:
    """Estimate a baseline from robust levels at both ends of a region."""

    x, y = _arrays(energy, intensity)
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must be within (0, 0.5].")
    left = _edge_level(y, left=True, fraction=edge_fraction, statistic=edge_statistic) if left_level is None else float(left_level)
    right = _edge_level(y, left=False, fraction=edge_fraction, statistic=edge_statistic) if right_level is None else float(right_level)
    return np.interp(x, (x[0], x[-1]), (left, right))


def shirley_background(
    energy: ArrayLike,
    intensity: ArrayLike,
    *,
    left_level: float | None = None,
    right_level: float | None = None,
    edge_fraction: float = 0.03,
    edge_statistic: EdgeStatistic | str = EdgeStatistic.MEDIAN,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
) -> NDArray[np.float64]:
    """Iteratively estimate a Shirley background.

    The endpoint levels are robust edge medians unless supplied. The returned
    curve always retains those endpoint levels, which makes its provenance
    reproducible and avoids silent extrapolation.
    """

    x, y = _arrays(energy, intensity)
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("max_iterations must be positive and tolerance must be greater than zero.")
    left = _edge_level(y, left=True, fraction=edge_fraction, statistic=edge_statistic) if left_level is None else float(left_level)
    right = _edge_level(y, left=False, fraction=edge_fraction, statistic=edge_statistic) if right_level is None else float(right_level)
    background = np.interp(x, (x[0], x[-1]), (left, right))

    for _ in range(max_iterations):
        net = np.clip(y - background, 0.0, None)
        increments = 0.5 * (net[1:] + net[:-1]) * np.diff(x)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        total = cumulative[-1]
        if total <= np.finfo(float).eps:
            return np.interp(x, (x[0], x[-1]), (left, right))

        remaining_fraction = (total - cumulative) / total
        candidate = right + (left - right) * remaining_fraction
        candidate[0], candidate[-1] = left, right
        if np.max(np.abs(candidate - background)) <= tolerance:
            return candidate
        background = candidate
    return background


def tougaard_background(
    energy: ArrayLike,
    intensity: ArrayLike,
    *,
    b: float = 2866.0,
    c: float = 1643.0,
    d: float = 0.0,
    scale: float | None = None,
    variant: TougaardVariant | str = TougaardVariant.U2,
    left_level: float | None = None,
    right_level: float | None = None,
    edge_fraction: float = 0.03,
    edge_statistic: EdgeStatistic | str = EdgeStatistic.LOWER_QUANTILE,
) -> NDArray[np.float64]:
    """Return an endpoint-anchored Tougaard-style inelastic-loss background.

    U2 uses B*T/(C+T**2)**2. U3 adds the positive D*T**2 denominator term.
    For an ascending binding-energy axis the loss tail is accumulated from
    lower to higher binding energy. Automatic scaling uses robust low-feature
    edge levels, so the background retains a physically inspectable baseline
    instead of collapsing to zero at one edge.

    A numeric scale is an explicit analyst constraint. With scale=None (the
    default), the kernel is scaled to the two measured edge anchors. Record
    B/C/D/scale and validate them against the laboratory method before
    reporting quantitative results.
    """

    x, y = _arrays(energy, intensity)
    selected_variant = TougaardVariant(variant)
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must be within (0, 0.5].")
    if b <= 0.0 or c <= 0.0 or d < 0.0 or (scale is not None and scale < 0.0):
        raise ValueError("Tougaard parameters b/c must be positive; d and scale cannot be negative.")

    left = (
        _edge_level(y, left=True, fraction=edge_fraction, statistic=edge_statistic)
        if left_level is None
        else float(left_level)
    )
    right = (
        _edge_level(y, left=False, fraction=edge_fraction, statistic=edge_statistic)
        if right_level is None
        else float(right_level)
    )

    # Subtract the low-binding-energy baseline before the loss convolution.
    # This avoids treating a constant detector offset as a loss tail.
    primary_signal = np.clip(y - left, 0.0, None)
    tail = np.zeros_like(y)
    for index in range(1, x.size):
        delta = x[index] - x[:index]
        denominator = np.square(c + np.square(delta))
        if selected_variant is TougaardVariant.U3:
            denominator = denominator + d * np.square(delta)
        kernel = b * delta / denominator
        tail[index] = np.trapezoid(primary_signal[:index] * kernel, x[:index])

    if scale is None:
        end_tail = tail[-1]
        if end_tail <= np.finfo(float).eps:
            return np.interp(x, (x[0], x[-1]), (left, right))
        scale_value = (right - left) / end_tail
    else:
        scale_value = float(scale)

    result = left + scale_value * tail
    result[0] = left
    return result


def custom_background(energy: ArrayLike, intensity: ArrayLike, *, values: ArrayLike) -> NDArray[np.float64]:
    """Validate a user-supplied/constrained background without hiding it."""

    x, _ = _arrays(energy, intensity)
    background = np.asarray(values, dtype=float).reshape(-1)
    if background.size != x.size or not np.all(np.isfinite(background)):
        raise ValueError("Custom background must be finite and match the spectrum length.")
    return background.copy()


def diagnose_background(spectrum: "Spectrum", background: ArrayLike) -> BackgroundDiagnostics:
    """Return transparent quality checks for a calculated background.

    The diagnostics are deliberately advisory: they flag background points
    noticeably above the measured trace but never silently replace a
    user-selected scientific model.
    """

    _, signal = _arrays(spectrum.energy, spectrum.intensity)
    values = np.asarray(background, dtype=float).reshape(-1)
    if values.size != signal.size or not np.all(np.isfinite(values)):
        raise ValueError("Background diagnostics require a finite, aligned background.")
    net = signal - values
    noise = max(float(spectrum.robust_noise()), np.finfo(float).eps)
    return BackgroundDiagnostics(
        left_anchor=float(values[0]),
        right_anchor=float(values[-1]),
        fraction_above_signal=float(np.mean(values > signal + 2.0 * noise)),
        minimum_net_intensity=float(np.min(net)),
        robust_net_noise=noise,
    )


def estimate_background(
    spectrum: "Spectrum",
    method: BackgroundMethod | str,
    **options: object,
) -> NDArray[np.float64]:
    """Estimate one named background for a :class:`~xps_afruz.Spectrum`."""

    selected = BackgroundMethod(method)
    if selected is BackgroundMethod.LINEAR:
        return linear_background(spectrum.energy, spectrum.intensity, **options)
    if selected is BackgroundMethod.SHIRLEY:
        return shirley_background(spectrum.energy, spectrum.intensity, **options)
    if selected is BackgroundMethod.TOUGAARD:
        return tougaard_background(spectrum.energy, spectrum.intensity, **options)
    if selected is BackgroundMethod.CUSTOM:
        return custom_background(spectrum.energy, spectrum.intensity, **options)
    raise AssertionError(f"Unhandled background method: {selected}")
