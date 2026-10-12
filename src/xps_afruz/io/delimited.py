"""Conservative CSV/TSV/two-column text import for Phase 1."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from ..core.spectrum import Spectrum
from ..core.energy import EnergyScale


def read_delimited_spectrum(
    path: str | Path,
    *,
    delimiter: str | None = None,
    label: str | None = None,
) -> Spectrum:
    """Read a two-column energy/intensity file with optional header rows.

    Lines beginning with ``#`` and nonnumeric header rows are ignored. This
    intentionally narrow importer is a dependable first step; vendor and VAMAS
    adapters belong in later phases rather than being guessed silently here.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Spectrum file was not found: {source}")
    rows: list[tuple[float, float]] = []
    header_lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header_lines.append(line[1:].strip())
            continue
        selected_delimiter = delimiter or _detect_delimiter(line)
        fields = line.split(selected_delimiter) if selected_delimiter else line.split()
        if len(fields) < 2:
            continue
        try:
            energy, intensity = float(fields[0].strip()), float(fields[1].strip())
        except ValueError:
            header_lines.append(line)
            continue
        rows.append((energy, intensity))
    if len(rows) < 4:
        raise ValueError("The file must contain at least four numeric energy/intensity rows.")
    values = np.asarray(rows, dtype=float)
    metadata = _metadata_from_headers(header_lines)
    metadata.update({"source_path": str(source), "format": "delimited", "header_lines": header_lines})
    return Spectrum(
        values[:, 0],
        values[:, 1],
        label=label or source.stem,
        metadata=metadata,
        energy_scale=_energy_scale_from_headers(header_lines),
        photon_energy_eV=_photon_energy_from_headers(header_lines),
    )


def _detect_delimiter(line: str) -> str | None:
    if "\t" in line:
        return "\t"
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    return None


def _metadata_from_headers(lines: list[str]) -> dict[str, object]:
    """Preserve common text-export acquisition fields without guessing units."""

    acquisition: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        normalized = key.strip().lower().replace(" ", "_")
        if normalized:
            acquisition[normalized] = value.strip()
    return {"acquisition": acquisition} if acquisition else {}


def _energy_scale_from_headers(lines: list[str]) -> EnergyScale:
    text = " ".join(lines).lower()
    if "kinetic energy" in text or "kinetic_energy" in text:
        return EnergyScale.KINETIC
    return EnergyScale.BINDING


def _photon_energy_from_headers(lines: list[str]) -> float | None:
    text = " ".join(lines)
    match = re.search(r"(?:photon|source)\s*energy\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    normalized = text.lower().replace(" ", "")
    if "al-k" in normalized or "alka" in normalized:
        return 1486.68
    if "mg-k" in normalized or "mgka" in normalized:
        return 1253.60
    return None
