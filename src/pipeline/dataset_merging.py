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


JSONL_OUTPUT = "final_dataset.jsonl"
CSV_OUTPUT = "final_dataset_summary.csv"


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

            mean_air = results["features"]["mean_air_temperature"]
            air = results["features"]["air_temperature"]

            row = {
                "home_id": os.path.basename(home),
                "inspection_note": note,
                **label,

                # Summary stats
                "mean_air_temp_avg": mean_air["average"],
                "mean_air_temp_min": mean_air["min"],
                "mean_air_temp_max": mean_air["max"],
                "air_temp_avg": air["average"],
                "air_temp_min": air["min"],
                "air_temp_max": air["max"],

                # Full hourly time series
                "mean_air_temp_hourly": mean_air["hourly"],
                "air_temp_hourly": air["hourly"]
            }

            rows.append(row)

        except Exception as e:
            print(f"[SKIP] Could not process {home}: {e}")

    with open(JSONL_OUTPUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary_rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    pd.DataFrame(summary_rows).to_csv(CSV_OUTPUT, index=False)