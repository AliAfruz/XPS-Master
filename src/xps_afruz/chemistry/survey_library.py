"""A transparent, curated XPS survey reference library.

The library intentionally contains common, diagnostic photoelectron lines
rather than pretending to reproduce a full chemical-state database.  Binding
energies are approximate screening positions and can shift with oxidation
state, charging, calibration, and instrumental conditions.  Each reported
match must be checked against the NIST XPS Database (SRD 20), laboratory
standards, and the full spectral context before it becomes an assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.energy import EnergyScale


@dataclass(frozen=True, slots=True)
class ReferenceLine:
    """One approximate, diagnostic XPS line for survey screening."""

    symbol: str
    element: str
    line: str
    binding_energy: float | None
    relative_strength: float = 1.0
    feature_type: str = "photoelectron"
    kinetic_energy: float | None = None
    screening_context: str = ""

    @property
    def label(self) -> str:
        """A compact label suitable for tables and annotations."""

        context = f" · {self.screening_context}" if self.screening_context else ""
        return f"{self.symbol} {self.line}{context}"

    def expected_energy(self, scale: EnergyScale | str, photon_energy_eV: float | None) -> float | None:
        """Return the library position on the supplied active energy scale.

        Photoelectron lines are tabulated as approximate binding energies;
        Auger lines are tabulated as kinetic energies.  Conversion is only
        possible when the photon energy is known explicitly.
        """

        selected_scale = EnergyScale(scale)
        feature = self.feature_type.lower()
        if feature == "photoelectron":
            if self.binding_energy is None:
                return None
            if selected_scale is EnergyScale.BINDING:
                return self.binding_energy
            return None if photon_energy_eV is None else photon_energy_eV - self.binding_energy
        if feature == "auger":
            if self.kinetic_energy is None:
                return None
            if selected_scale is EnergyScale.KINETIC:
                return self.kinetic_energy
            return None if photon_energy_eV is None else photon_energy_eV - self.kinetic_energy
        raise ValueError(f"Unsupported reference feature type: {self.feature_type!r}")


# Approximate, representative binding energies (eV).  Values are deliberately
# kept in source code so analysts can audit or extend the initial screening
# set.  They are not chemical-state assignments or quantification factors.
REFERENCE_LIBRARY: tuple[ReferenceLine, ...] = (
    ReferenceLine("Li", "Lithium", "1s", 55.5),
    ReferenceLine("Be", "Beryllium", "1s", 111.5),
    # B 1s can span a broad screening range across low- and high-binding-
    # energy environments. Keep the alternatives explicit so a narrow global
    # tolerance does not erase known weak B while avoiding a state assignment.
    ReferenceLine("B", "Boron", "1s", 187.5, screening_context="low-BE screening"),
    ReferenceLine("B", "Boron", "1s", 191.5, screening_context="high-BE screening"),
    ReferenceLine("C", "Carbon", "1s", 284.8),
    ReferenceLine("N", "Nitrogen", "1s", 399.8),
    ReferenceLine("O", "Oxygen", "1s", 531.0),
    ReferenceLine("F", "Fluorine", "1s", 685.0),
    ReferenceLine("Na", "Sodium", "1s", 1071.0),
    ReferenceLine("Mg", "Magnesium", "1s", 1303.0),
    ReferenceLine("Mg", "Magnesium", "2p", 49.5, 0.42),
    ReferenceLine("Al", "Aluminium", "2p", 74.5),
    ReferenceLine("Al", "Aluminium", "2s", 119.0, 0.48),
    ReferenceLine("Si", "Silicon", "2p", 99.4),
    ReferenceLine("Si", "Silicon", "2s", 150.5, 0.50),
    ReferenceLine("P", "Phosphorus", "2p", 133.5),
    ReferenceLine("S", "Sulfur", "2p", 164.0),
    ReferenceLine("Cl", "Chlorine", "2p", 200.2),
    ReferenceLine("K", "Potassium", "2p3/2", 293.2),
    ReferenceLine("Ca", "Calcium", "2p3/2", 347.0),
    ReferenceLine("Sc", "Scandium", "2p3/2", 401.5),
    ReferenceLine("Ti", "Titanium", "2p3/2", 458.7),
    ReferenceLine("V", "Vanadium", "2p3/2", 516.5),
    ReferenceLine("Cr", "Chromium", "2p3/2", 576.7),
    ReferenceLine("Mn", "Manganese", "2p3/2", 641.7),
    ReferenceLine("Fe", "Iron", "2p3/2", 710.8),
    ReferenceLine("Fe", "Iron", "2p1/2", 724.0, 0.50),
    ReferenceLine("Co", "Cobalt", "2p3/2", 780.0),
    ReferenceLine("Co", "Cobalt", "2p1/2", 795.0, 0.50),
    ReferenceLine("Ni", "Nickel", "2p3/2", 852.6),
    ReferenceLine("Ni", "Nickel", "2p1/2", 869.9, 0.50),
    ReferenceLine("Cu", "Copper", "2p3/2", 932.7),
    ReferenceLine("Cu", "Copper", "2p1/2", 952.5, 0.50),
    ReferenceLine("Zn", "Zinc", "2p3/2", 1021.8),
    ReferenceLine("Zn", "Zinc", "2p1/2", 1044.9, 0.50),
    ReferenceLine("Ga", "Gallium", "2p3/2", 1117.0),
    ReferenceLine("Ge", "Germanium", "3d", 29.4),
    ReferenceLine("As", "Arsenic", "3d", 44.5),
    ReferenceLine("Se", "Selenium", "3d", 54.5),
    ReferenceLine("Br", "Bromine", "3d", 70.0),
    ReferenceLine("Rb", "Rubidium", "3d", 110.0),
    ReferenceLine("Sr", "Strontium", "3d5/2", 133.5),
    ReferenceLine("Y", "Yttrium", "3d5/2", 157.0),
    ReferenceLine("Zr", "Zirconium", "3d5/2", 182.2),
    ReferenceLine("Nb", "Niobium", "3d5/2", 207.0),
    ReferenceLine("Mo", "Molybdenum", "3d5/2", 229.5),
    ReferenceLine("Ru", "Ruthenium", "3d5/2", 280.0),
    ReferenceLine("Rh", "Rhodium", "3d5/2", 307.0),
    ReferenceLine("Pd", "Palladium", "3d5/2", 335.0),
    ReferenceLine("Ag", "Silver", "3d5/2", 368.3),
    ReferenceLine("Cd", "Cadmium", "3d5/2", 405.0),
    ReferenceLine("In", "Indium", "3d5/2", 444.5),
    ReferenceLine("Sn", "Tin", "3d5/2", 486.5),
    ReferenceLine("Sb", "Antimony", "3d5/2", 528.5),
    ReferenceLine("Te", "Tellurium", "3d5/2", 573.0),
    ReferenceLine("I", "Iodine", "3d5/2", 619.0),
    ReferenceLine("Cs", "Caesium", "3d5/2", 724.0),
    ReferenceLine("Ba", "Barium", "3d5/2", 780.0),
    ReferenceLine("Ba", "Barium", "3d3/2", 795.3, 0.67),
    ReferenceLine("La", "Lanthanum", "3d5/2", 834.5),
    ReferenceLine("Ce", "Cerium", "3d5/2", 882.0),
    ReferenceLine("Hf", "Hafnium", "4f7/2", 16.8),
    ReferenceLine("Ta", "Tantalum", "4f7/2", 21.7),
    ReferenceLine("W", "Tungsten", "4f7/2", 35.5),
    ReferenceLine("Re", "Rhenium", "4f7/2", 40.5),
    ReferenceLine("Os", "Osmium", "4f7/2", 50.0),
    ReferenceLine("Ir", "Iridium", "4f7/2", 60.8),
    ReferenceLine("Pt", "Platinum", "4f7/2", 71.2),
    ReferenceLine("Au", "Gold", "4f7/2", 84.0),
    ReferenceLine("Hg", "Mercury", "4f7/2", 100.5),
    ReferenceLine("Tl", "Thallium", "4f7/2", 118.5),
    ReferenceLine("Pb", "Lead", "4f7/2", 138.6),
    ReferenceLine("Bi", "Bismuth", "4f7/2", 158.7),
    ReferenceLine("Th", "Thorium", "4f7/2", 334.0),
    ReferenceLine("U", "Uranium", "4f7/2", 380.0),
    # Auger features are kinetic-energy references. Their displayed binding
    # energy therefore depends on the recorded photon energy.
    ReferenceLine("C", "Carbon", "KLL", None, 0.45, feature_type="auger", kinetic_energy=272.0),
    ReferenceLine("O", "Oxygen", "KLL", None, 0.55, feature_type="auger", kinetic_energy=510.0),
    ReferenceLine("F", "Fluorine", "KLL", None, 0.35, feature_type="auger", kinetic_energy=656.0),
    ReferenceLine("Na", "Sodium", "KLL", None, 0.22, feature_type="auger", kinetic_energy=990.0),
    ReferenceLine("Mg", "Magnesium", "KLL", None, 0.25, feature_type="auger", kinetic_energy=1180.0),
)
