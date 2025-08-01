"""
dataset_merging.py

Merges processed dataset entries into unified JSONL and CSV files for analysis or model training.

Responsibilities:
- Reads each home's label.json, results.json, and inspection note.
- Merges into final_dataset.jsonl and final_dataset_summary.csv.
- Handles missing or incomplete folders gracefully.

Pipeline Context:
Final Step. Consumes outputs from postprocessing.py.

Outputs:
- final_dataset.jsonl
- final_dataset_summary.csv
"""

import os
import glob
import json
import pandas as pd

from config import BASE_RESULTS_DIR


JSONL_OUTPUT = f"{BASE_RESULTS_DIR}/final_dataset.jsonl"
CSV_OUTPUT = f"{BASE_RESULTS_DIR}/final_dataset_summary.csv"


def merge_dataset(dataset_dir: str = "dataset") -> None:
    """
    Merges processed dataset entries into final JSONL and CSV outputs.

    Args:
        dataset_dir (str): Path to the dataset directory.
    """
    rows = []
    for home in glob.glob(os.path.join(dataset_dir, "*")):
        try:
            label = json.load(open(os.path.join(home, "label.json")))
            results = json.load(open(os.path.join(home, "results.json")))
            note = json.load(open(os.path.join(home, "cleaned.geojson")))["features"][0]["properties"]["inspection_note"]

            air = results["features"]["air_temperature"]
            heating = results["features"]["heating_coil"]
            facility = results["features"]["facility_electricity"]
            hvac = results["features"]["hvac_electricity"]

            row = {
                "home_id": os.path.basename(home),
                "inspection_note": note,
                **label,

                # Summary stats
                "air_temp_avg": air["average"],
                "air_temp_min": air["min"],
                "air_temp_max": air["max"],

                "heating_coil_avg": heating["average"],
                "heating_coil_min": heating["min"],
                "heating_coil_max": heating["max"],

                "facility_electricity_avg": facility["average"],
                "facility_electricity_min": facility["min"],
                "facility_electricity_max": facility["max"],

                "hvac_electricity_avg": hvac["average"],
                "hvac_electricity_min": hvac["min"],
                "hvac_electricity_max": hvac["max"],

                # Full hourly time series
                "air_temp_hourly": air["hourly"],
                "heating_coil_hourly": heating["hourly"],
                "facility_electricity_hourly": facility["hourly"],
                "hvac_electricity_hourly": hvac["hourly"],
            }

            rows.append(row)

        except Exception as e:
            print(f"[SKIP] Could not process {home}: {e}")

    with open(JSONL_OUTPUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary_rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    pd.DataFrame(summary_rows).to_csv(CSV_OUTPUT, index=False)
