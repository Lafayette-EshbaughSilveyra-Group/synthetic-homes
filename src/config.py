"""
Global configuration settings for Urban Energy Data Pipeline.

This module centralizes constants and configuration parameters used across
pipeline components and experiments.
"""

# File paths and directories
OUTPUT_DIR = "dataset"
BASE_RESULTS_DIR = "results"

# EnergyPlus simulation parameters
WEATHER_STATION = "KABE"  # Allentown/Bethlehem/Easton Airport

# Model parameters
DEFAULT_WALL_R_VALUE = 13.0
DEFAULT_ROOF_R_VALUE = 30.0
DEFAULT_WINDOW_U_VALUE = 2.0
DEFAULT_HEATING_COP = 0.8
DEFAULT_COOLING_COP = 3.0

# Energy thresholds for labeling
HL_WORST = 6000  # kWh/yr - Moved from postprocessing.py
HL_BEST = 25     # kWh/yr
HVAC_WORST = 3000  # kWh/yr
HVAC_BEST = 25     # kWh/yr

# Streets to scrape - Moved from main.py
STREETS = [
    "STANBRIDGE CT",
    "IRONSTONE RD",
    "IRONSTONE CT",
    "HIGHBRIDGE CT",
    "TUDOR CT",
    "SUTTON PL",
    "REGAL RD",
    "GRAMERCY PL",
    "MARGATE RD",
    "RAMBEAU RD",
    "CANTERBURY RD",
    "GLOUCESTER DR",
    "NIJARO RD"
]