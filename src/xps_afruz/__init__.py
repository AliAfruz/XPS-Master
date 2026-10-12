"""XPS-master scientific analysis core.

The public API deliberately separates data, baseline estimation, line shapes,
and fitting so an interactive application can be added without changing the
scientific fitting layer.
"""

from .metadata import APP_VERSION as __version__, AUTHOR_NAME as __author__
from .metadata import AUTHOR_EMAIL as __email__, AUTHOR_WEBSITE as __url__

from .backgrounds import (
    BackgroundDiagnostics,
    BackgroundMethod,
    BackgroundSettings,
    EdgeStatistic,
    TougaardVariant,
    diagnose_background,
    estimate_background,
)
from .chemistry.quantification import GENERIC_ALKA_RSF, SensitivityFactor, quantify_survey
from .chemistry.survey import SurveyAnalysis, analyze_survey
from .chemistry.survey_library import REFERENCE_LIBRARY, ReferenceLine
from .core.energy import EnergyCalibration, EnergyScale, charge_correct, convert_energy_scale
from .core.processing import (
    combine_spectra,
    derivative_spectrum,
    despike_spectrum,
    gaussian_smooth,
    normalize_spectrum,
    savitzky_golay_smooth,
    signal_to_noise,
    subtract_spectra,
)
from .core.spectrum import Spectrum
from .fitting.model import (
    FitModel,
    FitResult,
    ParameterTie,
    PeakComponent,
    make_spin_orbit_doublet,
)
from .fitting.constraint_templates import SPIN_ORBIT_TEMPLATES, SpinOrbitTemplate, spin_orbit_template
from .fitting.model_comparison import (
    ModelCandidate,
    ModelCandidateEvaluation,
    ModelComparisonCancelled,
    ModelComparisonResult,
    ModelComparisonSettings,
    build_standard_candidates,
    compare_models,
)
from .fitting.optimizer import fit_spectrum
from .fitting.global_series import (
    GlobalComponentTrend,
    GlobalSeriesCancelled,
    GlobalSeriesResult,
    GlobalSeriesSettings,
    GlobalSpectrumFit,
    fit_global_series,
    shift_global_series_energy,
)
from .fitting.peak_shapes import PeakProfile, ProfileFamily
from .fitting.seed_protection import SeedProtectionDiagnostics, diagnose_seeded_fit, protect_manual_peak_seeds
from .fitting.smart_deconvolution import SmartDeconvolutionResult, SmartDeconvolutionSettings, smart_deconvolve
from .fitting.stability_consensus import (
    ComponentStability,
    StabilityConsensusCancelled,
    StabilityConsensusResult,
    StabilityConsensusSettings,
    stability_consensus,
)

__all__ = [
    "BackgroundMethod",
    "BackgroundDiagnostics",
    "BackgroundSettings",
    "EdgeStatistic",
    "EnergyCalibration",
    "EnergyScale",
    "FitModel",
    "FitResult",
    "ParameterTie",
    "PeakComponent",
    "PeakProfile",
    "ProfileFamily",
    "SPIN_ORBIT_TEMPLATES",
    "SpinOrbitTemplate",
    "spin_orbit_template",
    "ModelCandidate",
    "ModelCandidateEvaluation",
    "ModelComparisonCancelled",
    "ModelComparisonResult",
    "ModelComparisonSettings",
    "build_standard_candidates",
    "compare_models",
    "REFERENCE_LIBRARY",
    "ReferenceLine",
    "Spectrum",
    "SmartDeconvolutionResult",
    "SmartDeconvolutionSettings",
    "ComponentStability",
    "StabilityConsensusCancelled",
    "StabilityConsensusResult",
    "StabilityConsensusSettings",
    "SurveyAnalysis",
    "SensitivityFactor",
    "analyze_survey",
    "charge_correct",
    "convert_energy_scale",
    "combine_spectra",
    "derivative_spectrum",
    "despike_spectrum",
    "diagnose_background",
    "estimate_background",
    "fit_spectrum",
    "SeedProtectionDiagnostics",
    "diagnose_seeded_fit",
    "protect_manual_peak_seeds",
    "GlobalComponentTrend",
    "GlobalSeriesCancelled",
    "GlobalSeriesResult",
    "GlobalSeriesSettings",
    "GlobalSpectrumFit",
    "fit_global_series",
    "shift_global_series_energy",
    "make_spin_orbit_doublet",
    "gaussian_smooth",
    "GENERIC_ALKA_RSF",
    "quantify_survey",
    "normalize_spectrum",
    "savitzky_golay_smooth",
    "signal_to_noise",
    "smart_deconvolve",
    "stability_consensus",
    "subtract_spectra",
    "TougaardVariant",
]
