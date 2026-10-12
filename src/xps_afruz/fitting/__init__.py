"""Peak models, physical ties, and numerical optimization."""

from .model import (
    FitModel,
    FitResult,
    ParameterTie,
    PeakComponent,
    make_spin_orbit_doublet,
    shift_fit_result_energy,
    shift_parameter_ties_energy,
)
from .constraint_templates import SPIN_ORBIT_TEMPLATES, SpinOrbitTemplate, spin_orbit_template
from .model_comparison import (
    ModelCandidate,
    ModelCandidateEvaluation,
    ModelComparisonCancelled,
    ModelComparisonResult,
    ModelComparisonSettings,
    build_standard_candidates,
    compare_models,
    optimized_components,
    shift_model_comparison_energy,
)
from .optimizer import fit_spectrum
from .peak_shapes import PeakProfile, ProfileFamily, evaluate_peak
from .seed_protection import SeedProtectionDiagnostics, diagnose_seeded_fit, protect_manual_peak_seeds
from .global_series import (
    GlobalComponentTrend,
    GlobalSeriesCancelled,
    GlobalSeriesResult,
    GlobalSeriesSettings,
    GlobalSpectrumFit,
    fit_global_series,
    shift_global_series_energy,
)
from .smart_deconvolution import (
    SmartDeconvolutionCancelled,
    SmartDeconvolutionResult,
    SmartDeconvolutionSettings,
    smart_deconvolve,
)
from .stability_consensus import (
    ComponentStability,
    StabilityConsensusCancelled,
    StabilityConsensusResult,
    StabilityConsensusSettings,
    stability_consensus,
)

__all__ = [
    "FitModel",
    "FitResult",
    "ParameterTie",
    "PeakComponent",
    "PeakProfile",
    "ProfileFamily",
    "evaluate_peak",
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
    "shift_fit_result_energy",
    "shift_parameter_ties_energy",
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
    "optimized_components",
    "shift_model_comparison_energy",
    "SmartDeconvolutionCancelled",
    "SmartDeconvolutionResult",
    "SmartDeconvolutionSettings",
    "smart_deconvolve",
    "ComponentStability",
    "StabilityConsensusCancelled",
    "StabilityConsensusResult",
    "StabilityConsensusSettings",
    "stability_consensus",
]
