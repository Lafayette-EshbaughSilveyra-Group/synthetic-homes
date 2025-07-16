from . import text_variation
from . import energyplus_variation
from . import combined_input_variation
from . import plot_results
from . import run_experiments

HVAC_TEXT_SAMPLES = [
    # Very bad (very inefficient, needs replacement)
    "There is an older HVAC unit installed, with signs of rust on the exterior.",
    # Bad (inefficient, functional but outdated)
    "The HVAC system appears to be in working condition but is an older standard-efficiency model.",
    # Medium (moderate efficiency)
    "The home uses window AC units rather than a central HVAC system.",
    # Good (efficient)
    "The HVAC system was recently replaced with a standard high-efficiency model and is expected to operate efficiently.",
    # Very good (very efficient, top-tier system)
    "A state-of-the-art HVAC system with smart thermostats and variable-speed compressors was recently installed, maximizing energy efficiency."
]

INSULATION_TEXT_SAMPLES = [
    # Very bad (very inefficient, minimal insulation)
    "Attic insulation is minimal, with exposed joists visible throughout, causing significant heat loss.",
    # Bad (poor insulation)
    "No signs of added insulation were observed in the basement ceiling, suggesting potential energy inefficiency.",
    # Medium (moderate insulation)
    "Walls appear to be adequately insulated based on construction year, though no upgrades were observed.",
    # Good (efficient insulation)
    "Blown-in insulation is present in the attic to a depth of approximately 10 inches, providing good thermal resistance.",
    # Very good (very efficient, top-tier insulation)
    "High-performance spray foam insulation was installed throughout the walls, attic, and basement, providing maximum energy efficiency."
]

__all__ = [
    "text_variation",
    "energyplus_variation",
    "combined_input_variation",
    "plot_results",
    "run_experiments",
    "HVAC_TEXT_SAMPLES",
    "INSULATION_TEXT_SAMPLES"
]
