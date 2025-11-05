#!/bin/bash

cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

# Color definitions
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
  echo "Urban Energy Data Pipeline"
  echo "(c) 2025 Eshbaugh et al"
  echo "============================"
  echo "What would you like to run?"
  echo
  echo "1) Run full pipeline (scrape → GeoJSON → IDF → EnergyPlus → labels)"
  echo "2) Run pipeline except for scraping (GeoJSON → IDF → EnergyPlus → labels)"
  echo "3) Run labeling experiments"
  echo "4) Run occlusion experiments"
  echo "5) Generate factorial data (used to build scalers)"
  echo "6) Build scalers (used in the labeler)"
  echo "7) Exit"
  echo

  read -p "Enter choice [1-7]: " choice

  case $choice in
    1)
      echo "Running full data generation pipeline..."
      python3 src/main.py --mode pipeline
      ;;
    2)
      echo "Running pipeline without scraping..."
      python3 src/main.py --mode pipeline-no-scrape
      ;;
    3)
      echo "Running labeling experiments..."
      python3 src/main.py --mode experiments
      ;;
    4)
      echo "Running occlusion experiments..."
      python3 src/main.py --mode occlusion
      ;;
    5)
      echo "Generating factorial dataset..."
      python3 src/main.py --mode generate-factorial
      ;;
    6)
      echo "Building scalers..."
      python3 src/main.py --mode build-scalers
      ;;
    7)
      echo "Exiting."
      break
      ;;
    *)
      echo -e "${RED}Invalid choice.${NC} Please run again and choose a value between 1–5."
      exit 1
      ;;
  esac
done