"""
Global configuration settings for Urban Energy Data Pipeline.

This module centralizes constants and configuration parameters used across
pipeline components and experiments.
"""

# File paths and directories
OUTPUT_DIR = "dataset"
BASE_RESULTS_DIR = "results"

# EnergyPlus simulation parameters

# Ensure you have placed the three weather files of the same name (minus extensions) into the weather/ folder.
# For example, KABE.epw, KABE.stat, and KABE.ddy are located in weather/.
WEATHER_STATION = "KABE"  # Allentown/Bethlehem/Easton Airport

IDD_FILE_PATH = "/usr/local/EnergyPlus-25-1-0/Energy+.idd"

# Model parameters
DEFAULT_WALL_R_VALUE = 13.0
DEFAULT_ROOF_R_VALUE = 30.0
DEFAULT_WINDOW_U_VALUE = 2.0
DEFAULT_HEATING_COP = 0.8
DEFAULT_COOLING_COP = 3.0

# Streets to scrape
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