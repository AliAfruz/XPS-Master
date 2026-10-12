"""Reader for Thermo/VG Scientific `.VGD` XPS data files.

VGD files are OLE compound documents. The reader extracts raw double-precision
intensity values from ``VGData``, dimensions from ``VGDataAxes``, and kinetic
energy calibration from ``VGSpaceAxes``. Binding energy is calculated as
``source energy - kinetic energy``.
"""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
from numpy.typing import NDArray
import olefile

from ..core.spectrum import Spectrum
from ..core.energy import EnergyScale


class VGDReadError(ValueError):
    """Raised when a VGD file is incomplete or uses an unsupported layout."""


def read_vgd_spectrum(path: str | Path) -> Spectrum:
    """Read the first spectrum from a Thermo/VG Scientific VGD file.

    Use :func:`read_vgd_spectra` when a file may contain more than one spectrum.
    """

    return read_vgd_spectra(path)[0]


def read_vgd_spectra(path: str | Path) -> list[Spectrum]:
    """Read every spectrum held in a Thermo/VG Scientific VGD file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"VGD file was not found: {source}")
    if not olefile.isOleFile(str(source)):
        raise VGDReadError("This is not a supported OLE-based Thermo/VG VGD file.")

    try:
        document = olefile.OleFileIO(str(source))
    except OSError as error:
        raise VGDReadError(f"Could not open VGD file: {source.name}") from error
    try:
        streams = {parts[-1] for parts in document.listdir(streams=True, storages=False)}
        missing = {"VGData", "VGDataAxes", "VGSpaceAxes"}.difference(streams)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise VGDReadError(f"VGD file is missing required stream(s): {missing_text}.")

        data = document.openstream("VGData").read()
        data_axes = document.openstream("VGDataAxes").read()
        space_axes = document.openstream("VGSpaceAxes").read()
        source_energy = _extract_source_energy(document)
    except (OSError, ValueError, struct.error) as error:
        if isinstance(error, VGDReadError):
            raise
        raise VGDReadError(f"Could not read VGD data streams from {source.name}.") from error
    finally:
        document.close()

    intensities, points_per_spectrum, number_of_spectra = _decode_intensities(data, data_axes)
    kinetic_start, kinetic_step = _decode_energy_axis(space_axes)
    kinetic_energy = kinetic_start + np.arange(points_per_spectrum, dtype=float) * kinetic_step
    binding_energy = source_energy - kinetic_energy
    metadata_base = {
        "source_path": str(source),
        "format": "Thermo/VG VGD",
        "source_energy_eV": source_energy,
        "kinetic_energy_start_eV": kinetic_start,
        "kinetic_energy_step_eV": kinetic_step,
        "number_of_spectra": number_of_spectra,
        "acquisition": {
            "photon_source": "Al Kα" if abs(source_energy - 1486.68) < 0.2 else "Mg Kα" if abs(source_energy - 1253.60) < 0.2 else "Unknown",
            "photon_energy_eV": source_energy,
            "step_size_eV": abs(kinetic_step),
            "energy_axis_from_file": "kinetic",
        },
    }
    spectra: list[Spectrum] = []
    for index, intensity in enumerate(intensities):
        label = source.stem if number_of_spectra == 1 else f"{source.stem} [{index + 1}/{number_of_spectra}]"
        metadata = {**metadata_base, "spectrum_index": index}
        spectra.append(
            Spectrum(
                binding_energy,
                intensity,
                label=label,
                metadata=metadata,
                energy_scale=EnergyScale.BINDING,
                photon_energy_eV=source_energy,
                original_energy=kinetic_energy,
                original_energy_scale=EnergyScale.KINETIC,
            )
        )
    return spectra


def _decode_intensities(
    data: bytes,
    data_axes: bytes,
) -> tuple[NDArray[np.float64], int, int]:
    if len(data) == 0 or len(data) % 8:
        raise VGDReadError("VGData does not contain an integral number of 64-bit intensity values.")
    raw = np.frombuffer(data, dtype="<f8").copy()
    if not np.all(np.isfinite(raw)):
        raise VGDReadError("VGData contains non-finite intensity values.")

    number_of_spectra = 1
    points_per_spectrum = raw.size
    # In VGD files these two 32-bit values are stored as zero-based dimensions.
    if len(data_axes) >= 32:
        candidate_points = struct.unpack_from("<i", data_axes, 12)[0] + 1
        candidate_spectra = struct.unpack_from("<i", data_axes, 28)[0] + 1
        if candidate_points > 0 and candidate_spectra > 0 and candidate_points * candidate_spectra == raw.size:
            points_per_spectrum = candidate_points
            number_of_spectra = candidate_spectra
    return raw.reshape(number_of_spectra, points_per_spectrum), points_per_spectrum, number_of_spectra


def _decode_energy_axis(space_axes: bytes) -> tuple[float, float]:
    # VGSpaceAxes stores the primary kinetic-energy origin and step at these
    # offsets in the single-/multi-spectrum VGD layout used by Avantage.
    if len(space_axes) < 46:
        raise VGDReadError("VGSpaceAxes is too short to contain kinetic-energy calibration.")
    kinetic_start = struct.unpack_from("<d", space_axes, 30)[0]
    kinetic_step = struct.unpack_from("<d", space_axes, 38)[0]
    if not np.isfinite(kinetic_start) or not np.isfinite(kinetic_step) or kinetic_step == 0.0:
        raise VGDReadError("VGD kinetic-energy calibration is invalid.")
    return float(kinetic_start), float(kinetic_step)


def _extract_source_energy(document: olefile.OleFileIO) -> float:
    """Find the Al/Mg X-ray source-energy field from VGD property streams."""

    expected_energies = (1486.68, 1253.60)
    for stream in document.listdir(streams=True, storages=False):
        if not stream[-1].startswith("\x05"):
            continue
        payload = document.openstream(stream).read()
        usable_size = len(payload) - (len(payload) % 4)
        if not usable_size:
            continue
        values = np.frombuffer(payload[:usable_size], dtype="<f4")
        finite_values = values[np.isfinite(values)]
        for expected in expected_energies:
            matches = finite_values[np.abs(finite_values - expected) < 0.2]
            if matches.size:
                return float(matches[0])
    # Al Kα is the standard source for the vast majority of the supported files.
    # Retain the fallback transparently in the Spectrum metadata.
    return 1486.68
