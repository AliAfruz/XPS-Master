"""Reusable orbital-degeneracy templates for explicit spin-orbit constraints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpinOrbitTemplate:
    """A generic degeneracy ratio; experimental splitting remains user supplied."""

    key: str
    label: str
    upper_to_lower_area_ratio: float
    lower_j: str
    upper_j: str
    note: str


SPIN_ORBIT_TEMPLATES: tuple[SpinOrbitTemplate, ...] = (
    SpinOrbitTemplate(
        "custom",
        "Custom ratio",
        0.5,
        "lower BE",
        "higher BE",
        "Enter a laboratory-justified ratio and splitting.",
    ),
    SpinOrbitTemplate(
        "p",
        "p doublet — 2p3/2 : 2p1/2 = 2:1",
        0.5,
        "j=3/2",
        "j=1/2",
        "Statistical degeneracy ratio for an ideal p doublet; verify material-specific deviations.",
    ),
    SpinOrbitTemplate(
        "d",
        "d doublet — 3d5/2 : 3d3/2 = 3:2",
        2.0 / 3.0,
        "j=5/2",
        "j=3/2",
        "Statistical degeneracy ratio for an ideal d doublet; verify material-specific deviations.",
    ),
    SpinOrbitTemplate(
        "f",
        "f doublet — 4f7/2 : 4f5/2 = 4:3",
        0.75,
        "j=7/2",
        "j=5/2",
        "Statistical degeneracy ratio for an ideal f doublet; verify material-specific deviations.",
    ),
)


def spin_orbit_template(key: str) -> SpinOrbitTemplate:
    """Return one named template without guessing an elemental assignment."""

    for template in SPIN_ORBIT_TEMPLATES:
        if template.key == key:
            return template
    raise KeyError(key)
