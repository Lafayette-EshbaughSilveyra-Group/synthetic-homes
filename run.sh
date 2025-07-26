#!/bin/bash


cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies if not already installed
if [ ! -f "venv/.deps_installed" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt && touch venv/.deps_installed
fi

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
  echo "5) Exit"
  echo

  read -p "Enter choice [1-5]: " choice

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
      echo "Exiting."
      break
      ;;
    *)
      echo -e "${RED}Invalid choice.${NC} Please run again and choose a value between 1–5."
      exit 1
      ;;
  esac
done