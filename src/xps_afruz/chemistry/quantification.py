"""Transparent survey-level RSF quantification helpers.

The default factors are a generic Al Kα screening set normalized to C 1s = 1.
They make the model executable and auditable, but they are *not* a substitute
for an instrument-specific sensitivity and transmission calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .survey import SurveyAnalysis, SurveyMatch


@dataclass(frozen=True, slots=True)
class SensitivityFactor:
    """One effective factor and its declared correction model."""

    symbol: str
    line: str
    value: float
    instrument: str = "Generic Al Kα screening"
    source: str = "Editable generic RSF library, normalized to C 1s = 1"
    transmission_factor: float = 1.0
    analyzer_factor: float = 1.0
    relative_uncertainty: float = 0.15
    model: str = "relative_sensitivity_factor"

    def __post_init__(self) -> None:
        if self.value <= 0.0 or self.transmission_factor <= 0.0 or self.analyzer_factor <= 0.0:
            raise ValueError("Sensitivity and correction factors must be positive.")
        if self.relative_uncertainty < 0.0:
            raise ValueError("RSF relative uncertainty must be non-negative.")

    @property
    def effective_factor(self) -> float:
        """Combined denominator used by the documented correction model."""

        return self.value * self.transmission_factor * self.analyzer_factor


# Generic, editable starting factors for common Al Kα photoelectron lines.
# The library intentionally names its model and normalisation so results never
# look instrument-independent.  Add a user/laboratory library for reporting.
GENERIC_ALKA_RSF: tuple[SensitivityFactor, ...] = (
    SensitivityFactor("C", "1s", 1.000),
    SensitivityFactor("N", "1s", 1.800),
    SensitivityFactor("O", "1s", 2.930),
    SensitivityFactor("F", "1s", 4.430),
    SensitivityFactor("Na", "1s", 7.140),
    SensitivityFactor("Mg", "1s", 5.640),
    SensitivityFactor("Mg", "2p", 0.280),
    SensitivityFactor("Al", "2p", 0.537),
    SensitivityFactor("Si", "2p", 0.817),
    SensitivityFactor("P", "2p", 1.319),
    SensitivityFactor("S", "2p", 1.668),
    SensitivityFactor("Cl", "2p", 1.797),
    SensitivityFactor("K", "2p3/2", 1.999),
    SensitivityFactor("Ca", "2p3/2", 2.706),
    SensitivityFactor("Ti", "2p3/2", 2.001),
    SensitivityFactor("Cr", "2p3/2", 2.427),
    SensitivityFactor("Mn", "2p3/2", 2.930),
    SensitivityFactor("Fe", "2p3/2", 2.957),
    SensitivityFactor("Co", "2p3/2", 3.590),
    SensitivityFactor("Ni", "2p3/2", 4.100),
    SensitivityFactor("Cu", "2p3/2", 5.321),
    SensitivityFactor("Zn", "2p3/2", 6.000),
)


def load_sensitivity_factor_library(path: str | Path) -> tuple[SensitivityFactor, ...]:
    """Load an instrument/laboratory RSF JSON library for reproducible use.

    The file may be either a list of factor objects or an object containing a
    ``factors`` list.  Required fields are ``symbol``, ``line``, and ``value``;
    all correction/provenance fields use the :class:`SensitivityFactor`
    defaults when omitted.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read RSF library: {source}") from error
    entries = payload.get("factors") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("An RSF library must contain a list of factor objects.")
    try:
        factors = tuple(SensitivityFactor(**entry) for entry in entries if isinstance(entry, dict))
    except (TypeError, ValueError) as error:
        raise ValueError("RSF library contains an invalid factor.") from error
    if not factors or len(factors) != len(entries):
        raise ValueError("An RSF library must contain at least one valid factor object.")
    return factors


def save_sensitivity_factor_library(path: str | Path, factors: Iterable[SensitivityFactor]) -> None:
    """Write an auditable, human-readable RSF library JSON file."""

    target = Path(path)
    entries = [asdict(factor) for factor in factors]
    if not entries:
        raise ValueError("Refusing to save an empty RSF library.")
    target.write_text(json.dumps({"format": "xps-afruz-rsf-library", "factors": entries}, indent=2), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class QuantificationRecord:
    """A single elemental contribution to an atomic-percent calculation."""

    symbol: str
    element: str
    line: str
    peak_energy: float
    area: float
    sensitivity_factor: SensitivityFactor
    corrected_intensity: float
    atomic_percent: float
    atomic_percent_uncertainty: float
    signal_to_noise: float
    targeted_trace: bool


@dataclass(frozen=True, slots=True)
class CompositionResult:
    """An auditable composition calculation from selected survey matches."""

    records: tuple[QuantificationRecord, ...]
    excluded_elements: tuple[str, ...]
    unquantified_elements: tuple[str, ...]
    factor_model: str

    def as_percentages(self) -> dict[str, float]:
        """Return compact element → atomic-percent mapping."""

        return {record.symbol: record.atomic_percent for record in self.records}


@dataclass(frozen=True, slots=True)
class CompositionComparison:
    """Atomic-percent comparison table across independently processed samples."""

    sample_labels: tuple[str, ...]
    elements: tuple[str, ...]
    values: Mapping[str, Mapping[str, float]]


def _factor_index(factors: Iterable[SensitivityFactor]) -> dict[tuple[str, str], SensitivityFactor]:
    return {(factor.symbol.lower(), factor.line.lower()): factor for factor in factors}


def _choose_quantifiable_match(matches: Sequence[SurveyMatch], factor_index: Mapping[tuple[str, str], SensitivityFactor]) -> tuple[SurveyMatch, SensitivityFactor] | None:
    ranked = sorted(matches, key=lambda match: (match.reference.feature_type == "photoelectron", match.score), reverse=True)
    for match in ranked:
        factor = factor_index.get((match.reference.symbol.lower(), match.reference.line.lower()))
        if factor is not None and match.peak.background_corrected_area > 0.0:
            return match, factor
    return None


def quantify_survey(
    analysis: SurveyAnalysis,
    *,
    factors: Iterable[SensitivityFactor] = GENERIC_ALKA_RSF,
    factor_overrides: Mapping[tuple[str, str], float] | None = None,
    include_elements: Iterable[str] | None = None,
) -> CompositionResult:
    """Calculate atomic percent from locally background-corrected survey areas.

    ``corrected_intensity = area / (RSF × transmission × analyzer)`` and the
    atomic percent is the normalized corrected intensity.  Overrides replace
    only the RSF value while retaining the documented correction model.
    """

    index = _factor_index(factors)
    if factor_overrides:
        for raw_key, value in factor_overrides.items():
            if len(raw_key) != 2:
                raise ValueError("RSF override keys must be (element, line) pairs.")
            key = (str(raw_key[0]).lower(), str(raw_key[1]).lower())
            if key in index:
                original = index[key]
                index[key] = SensitivityFactor(
                    symbol=original.symbol,
                    line=original.line,
                    value=float(value),
                    instrument=original.instrument,
                    source=f"User override of {original.source}",
                    transmission_factor=original.transmission_factor,
                    analyzer_factor=original.analyzer_factor,
                    relative_uncertainty=original.relative_uncertainty,
                    model=original.model,
                )
            else:
                index[key] = SensitivityFactor(
                    symbol=str(raw_key[0]),
                    line=str(raw_key[1]),
                    value=float(value),
                    instrument="User-declared instrument factor",
                    source="User-defined RSF for a line absent from the generic library",
                    relative_uncertainty=0.20,
                    model="user_defined_rsf",
                )
    included = None if include_elements is None else {str(symbol).strip().lower() for symbol in include_elements}
    selected_candidates = [
        candidate
        for candidate in analysis.candidates
        if included is None or candidate.symbol.lower() in included
    ]
    unquantified: list[str] = []
    raw_records: list[tuple[SurveyMatch, SensitivityFactor, float]] = []
    for candidate in selected_candidates:
        selected = _choose_quantifiable_match(candidate.matches, index)
        if selected is None:
            unquantified.append(candidate.symbol)
            continue
        match, factor = selected
        corrected = match.peak.background_corrected_area / factor.effective_factor
        raw_records.append((match, factor, corrected))
    total = sum(record[2] for record in raw_records)
    if total <= 0.0:
        return CompositionResult((), analysis.excluded_elements, tuple(unquantified), "No quantifiable selected lines")
    records: list[QuantificationRecord] = []
    for match, factor, corrected in raw_records:
        atomic_percent = 100.0 * corrected / total
        area_relative_uncertainty = 1.0 / np.sqrt(max(match.peak.background_corrected_area, 1.0))
        if match.peak.targeted:
            # A local trace recovery carries a detection uncertainty that is
            # at least the inverse measured SNR, in addition to RSF error.
            area_relative_uncertainty = max(area_relative_uncertainty, 1.0 / max(match.peak.signal_to_noise, 1.0))
        relative_uncertainty = float(np.hypot(area_relative_uncertainty, factor.relative_uncertainty))
        records.append(
            QuantificationRecord(
                symbol=match.reference.symbol,
                element=match.reference.element,
                line=match.reference.line,
                peak_energy=match.peak.maximum_energy,
                area=match.peak.background_corrected_area,
                sensitivity_factor=factor,
                corrected_intensity=corrected,
                atomic_percent=atomic_percent,
                atomic_percent_uncertainty=atomic_percent * relative_uncertainty,
                signal_to_noise=match.peak.signal_to_noise,
                targeted_trace=match.peak.targeted,
            )
        )
    factor_models = sorted({record.sensitivity_factor.model for record in records})
    return CompositionResult(tuple(records), analysis.excluded_elements, tuple(unquantified), ", ".join(factor_models))


def compare_compositions(samples: Mapping[str, CompositionResult]) -> CompositionComparison:
    """Align atomic-percent results across named, independently analysed samples."""

    labels = tuple(samples)
    elements = tuple(sorted({record.symbol for result in samples.values() for record in result.records}))
    values = {
        symbol: {label: samples[label].as_percentages().get(symbol, 0.0) for label in labels}
        for symbol in elements
    }
    return CompositionComparison(labels, elements, values)
