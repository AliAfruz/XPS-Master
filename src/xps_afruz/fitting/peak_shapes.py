"""Area-normalized peak profiles used by the fitting engine."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz


class ProfileFamily(str, Enum):
    """Scientific interpretation category attached to a line-shape choice."""

    MATHEMATICAL = "mathematical"
    METALLIC_XPS = "metallic_xps"


class PeakProfile(str, Enum):
    """Available profiles with explicit mathematical or XPS-model provenance."""

    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    PSEUDO_VOIGT = "pseudo_voigt"
    GAUSSIAN_LORENTZIAN_PRODUCT = "gaussian_lorentzian_product"
    VOIGT = "voigt"
    ASYMMETRIC_PSEUDO_VOIGT = "asymmetric_pseudo_voigt"
    DONIACH_SUNJIC = "doniach_sunjic"

    @property
    def display_name(self) -> str:
        """Return a compact analyst-facing profile label."""

        labels = {
            PeakProfile.GAUSSIAN: "Gaussian (mathematical)",
            PeakProfile.LORENTZIAN: "Lorentzian (mathematical)",
            PeakProfile.PSEUDO_VOIGT: "G-L sum / pseudo-Voigt (mathematical)",
            PeakProfile.GAUSSIAN_LORENTZIAN_PRODUCT: "G-L product (mathematical)",
            PeakProfile.VOIGT: "Voigt convolution (mathematical)",
            PeakProfile.ASYMMETRIC_PSEUDO_VOIGT: "Asymmetric pseudo-Voigt (empirical)",
            PeakProfile.DONIACH_SUNJIC: "Doniach-Sunjic, metallic-state model",
        }
        return labels[self]

    @property
    def family(self) -> ProfileFamily:
        """Classify mathematical curves separately from the metallic-state model."""

        return ProfileFamily.METALLIC_XPS if self is PeakProfile.DONIACH_SUNJIC else ProfileFamily.MATHEMATICAL

    @property
    def scientific_note(self) -> str:
        """Return a short appropriate-use note for the selected shape."""

        notes = {
            PeakProfile.GAUSSIAN: "Mathematical broadening curve; it does not itself establish a chemical state.",
            PeakProfile.LORENTZIAN: "Mathematical lifetime-like curve; validate its use against acquisition conditions.",
            PeakProfile.PSEUDO_VOIGT: "Empirical Gaussian-Lorentzian sum. Eta is the Lorentzian fraction.",
            PeakProfile.GAUSSIAN_LORENTZIAN_PRODUCT: "Empirical Gaussian-Lorentzian product with a shared FWHM.",
            PeakProfile.VOIGT: "Gaussian-Lorentzian convolution. Eta controls the Lorentzian width contribution.",
            PeakProfile.ASYMMETRIC_PSEUDO_VOIGT: "Empirical asymmetric curve; positive alpha broadens the high-binding-energy side.",
            PeakProfile.DONIACH_SUNJIC: (
                "Finite-window Doniach-Sunjic-type metallic-state model. Alpha is the low-binding-energy tail parameter; "
                "instrument FWHM applies Gaussian broadening."
            ),
        }
        return notes[self]


def _validated_axis(energy: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(energy, dtype=float).reshape(-1)
    if x.size < 4 or not np.all(np.isfinite(x)):
        raise ValueError("Peak evaluation requires a one-dimensional finite energy axis with at least four points.")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("Peak evaluation requires a strictly ascending energy axis.")
    return x


def _validate_fwhm_and_area(fwhm: float, area: float) -> None:
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("FWHM must be finite and positive.")
    if not np.isfinite(area) or area < 0.0:
        raise ValueError("Peak area must be finite and non-negative.")


def _validate_eta(eta: float) -> None:
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("Eta must be finite and within [0, 1].")


def _normalize_finite_window(
    energy: NDArray[np.float64], raw: NDArray[np.float64], *, area: float
) -> NDArray[np.float64]:
    """Normalize a non-negative raw profile to the requested finite-window area."""

    if area == 0.0:
        return np.zeros_like(raw)
    clean = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    integral = float(np.trapezoid(clean, energy))
    if not np.isfinite(integral) or integral <= np.finfo(float).eps:
        raise ValueError("Peak shape has no finite positive area on the selected energy region.")
    return float(area) * clean / integral


def _validate_asymmetry(asymmetry: float, *, allow_negative: bool) -> None:
    lower = -0.95 if allow_negative else 0.0
    if not np.isfinite(asymmetry) or not lower <= asymmetry < 0.95:
        bounds = "[-0.95, 0.95)" if allow_negative else "[0, 0.95)"
        raise ValueError(f"Asymmetry must be finite and within {bounds}.")


def gaussian(energy: ArrayLike, *, center: float, fwhm: float, area: float) -> NDArray[np.float64]:
    """Area-normalized Gaussian with full width at half maximum ``fwhm``."""

    x = _validated_axis(energy)
    _validate_fwhm_and_area(fwhm, area)
    coefficient = float(area) * np.sqrt(4.0 * np.log(2.0) / np.pi) / fwhm
    return coefficient * np.exp(-4.0 * np.log(2.0) * np.square((x - center) / fwhm))


def lorentzian(energy: ArrayLike, *, center: float, fwhm: float, area: float) -> NDArray[np.float64]:
    """Area-normalized Lorentzian with full width at half maximum ``fwhm``."""

    x = _validated_axis(energy)
    _validate_fwhm_and_area(fwhm, area)
    half_width = 0.5 * fwhm
    return float(area) * half_width / (np.pi * (np.square(x - center) + half_width**2))


def pseudo_voigt(
    energy: ArrayLike,
    *,
    center: float,
    fwhm: float,
    area: float,
    eta: float = 0.5,
) -> NDArray[np.float64]:
    """Linear Gaussian/Lorentzian mixture with Lorentzian fraction ``eta``."""

    _validate_eta(eta)
    return (1.0 - eta) * gaussian(energy, center=center, fwhm=fwhm, area=area) + eta * lorentzian(
        energy, center=center, fwhm=fwhm, area=area
    )


def gaussian_lorentzian_product(
    energy: ArrayLike,
    *,
    center: float,
    fwhm: float,
    area: float,
) -> NDArray[np.float64]:
    """Return a finite-window normalized Gaussian-Lorentzian product.

    Its factors are scaled so the product reaches half maximum one half FWHM
    from the center, retaining an intuitive shared FWHM parameter.
    """

    x = _validated_axis(energy)
    _validate_fwhm_and_area(fwhm, area)
    relative = (x - center) / fwhm
    raw = np.exp(-2.0 * np.log(2.0) * np.square(relative)) / (
        1.0 + 4.0 * (np.sqrt(2.0) - 1.0) * np.square(relative)
    )
    return _normalize_finite_window(x, raw, area=area)


def voigt(
    energy: ArrayLike,
    *,
    center: float,
    fwhm: float,
    area: float,
    eta: float = 0.5,
) -> NDArray[np.float64]:
    """Return a finite-window normalized Gaussian-Lorentzian convolution.

    Eta is the Lorentzian width fraction. FWHM is the requested composite
    width, using the Olivero-Longbothum approximation for the input widths.
    """

    x = _validated_axis(energy)
    _validate_fwhm_and_area(fwhm, area)
    _validate_eta(eta)
    if eta == 0.0:
        return gaussian(x, center=center, fwhm=fwhm, area=area)
    if eta == 1.0:
        return lorentzian(x, center=center, fwhm=fwhm, area=area)
    lorentz_fwhm = eta * fwhm
    gaussian_squared = (fwhm - 0.5346 * lorentz_fwhm) ** 2 - 0.2166 * lorentz_fwhm**2
    gaussian_fwhm = np.sqrt(max(gaussian_squared, np.finfo(float).eps))
    sigma = gaussian_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = 0.5 * lorentz_fwhm
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    raw = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    return _normalize_finite_window(x, raw, area=area)


def asymmetric_pseudo_voigt(
    energy: ArrayLike,
    *,
    center: float,
    fwhm: float,
    area: float,
    eta: float = 0.5,
    asymmetry: float = 0.0,
) -> NDArray[np.float64]:
    """Return an empirical asymmetric pseudo-Voigt curve.

    Positive asymmetry broadens the high-binding-energy (right-hand,
    ascending-axis) side; it is not a chemical-state assignment.
    """

    x = _validated_axis(energy)
    _validate_fwhm_and_area(fwhm, area)
    _validate_eta(eta)
    _validate_asymmetry(asymmetry, allow_negative=True)
    delta = x - center
    local_fwhm = np.where(delta < 0.0, fwhm * (1.0 - asymmetry), fwhm * (1.0 + asymmetry))
    relative = delta / local_fwhm
    raw = (1.0 - eta) * np.exp(-4.0 * np.log(2.0) * np.square(relative)) + eta / (
        1.0 + 4.0 * np.square(relative)
    )
    return _normalize_finite_window(x, raw, area=area)


def doniach_sunjic(
    energy: ArrayLike,
    *,
    center: float,
    fwhm: float,
    area: float,
    asymmetry: float = 0.0,
    instrument_fwhm: float = 0.0,
) -> NDArray[np.float64]:
    """Return a finite-window Doniach-Sunjic-type metallic-state profile.

    Asymmetry is the usual non-negative alpha tail parameter. It produces a
    low-binding-energy tail on the internally ascending binding-energy axis.
    Instrument FWHM applies an explicit Gaussian broadening convolution.
    """

    x = _validated_axis(energy)
    _validate_fwhm_and_area(fwhm, area)
    _validate_asymmetry(asymmetry, allow_negative=False)
    if not np.isfinite(instrument_fwhm) or instrument_fwhm < 0.0:
        raise ValueError("Instrument FWHM must be finite and non-negative.")
    half_width = 0.5 * fwhm
    relative = (x - center) / half_width
    angle = np.arctan(relative)
    raw = np.cos(0.5 * np.pi * asymmetry + (1.0 - asymmetry) * angle) / np.power(
        1.0 + np.square(relative), 0.5 * (1.0 - asymmetry)
    )
    if instrument_fwhm > 0.0:
        raw = _gaussian_broaden(x, raw, instrument_fwhm)
    return _normalize_finite_window(x, raw, area=area)


def _gaussian_broaden(
    energy: NDArray[np.float64],
    values: NDArray[np.float64],
    instrument_fwhm: float,
) -> NDArray[np.float64]:
    """Convolve with an area-preserving Gaussian on uniform or interpolated axes."""

    step = float(np.median(np.diff(energy)))
    sigma_channels = instrument_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)) * step)
    if sigma_channels <= np.finfo(float).eps:
        return values
    differences = np.diff(energy)
    if np.allclose(differences, step, rtol=1e-4, atol=max(step * 1e-8, 1e-12)):
        return gaussian_filter1d(values, sigma_channels, mode="nearest")
    uniform = np.linspace(energy[0], energy[-1], energy.size)
    uniform_step = float(uniform[1] - uniform[0])
    uniform_sigma = instrument_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)) * uniform_step)
    broadened = gaussian_filter1d(np.interp(uniform, energy, values), uniform_sigma, mode="nearest")
    return np.interp(energy, uniform, broadened)


def evaluate_peak(
    energy: ArrayLike,
    *,
    profile: PeakProfile | str,
    center: float,
    fwhm: float,
    area: float,
    eta: float = 0.5,
    asymmetry: float = 0.0,
    instrument_fwhm: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate any named profile using an integrated-area parameter."""

    selected = PeakProfile(profile)
    if selected is PeakProfile.GAUSSIAN:
        return gaussian(energy, center=center, fwhm=fwhm, area=area)
    if selected is PeakProfile.LORENTZIAN:
        return lorentzian(energy, center=center, fwhm=fwhm, area=area)
    if selected is PeakProfile.PSEUDO_VOIGT:
        return pseudo_voigt(energy, center=center, fwhm=fwhm, area=area, eta=eta)
    if selected is PeakProfile.GAUSSIAN_LORENTZIAN_PRODUCT:
        return gaussian_lorentzian_product(energy, center=center, fwhm=fwhm, area=area)
    if selected is PeakProfile.VOIGT:
        return voigt(energy, center=center, fwhm=fwhm, area=area, eta=eta)
    if selected is PeakProfile.ASYMMETRIC_PSEUDO_VOIGT:
        return asymmetric_pseudo_voigt(
            energy, center=center, fwhm=fwhm, area=area, eta=eta, asymmetry=asymmetry
        )
    if selected is PeakProfile.DONIACH_SUNJIC:
        return doniach_sunjic(
            energy,
            center=center,
            fwhm=fwhm,
            area=area,
            asymmetry=asymmetry,
            instrument_fwhm=instrument_fwhm,
        )
    raise AssertionError(f"Unhandled peak profile: {selected}")
