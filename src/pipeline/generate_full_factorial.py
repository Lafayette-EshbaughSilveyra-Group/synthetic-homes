from __future__ import annotations
import json, os, glob, random, subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from eppy.modeleditor import IDF

from src import config
from src.pipeline.idf_generation import generate_idf_from_geojson

def _parse_eplusout_csvs_to_json(parent_folder: str, output_json_path: str):
    all_results = {}
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for folder in subfolders:
        csv_path = os.path.join(folder, "simulation_output", "eplusout.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {folder}, no simulation_output/eplusout.csv found.")
            continue

        df = pd.read_csv(csv_path)
        home_name = os.path.basename(folder)
        home_data = {}

        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                home_data[col] = {
                    "average": float(series.mean()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "hourly": [float(x) for x in series.tolist()]
                }

        all_results[home_name] = home_data

    with open(output_json_path, "w") as f:
        json.dump(all_results, f, indent=2)


def _generate_summary_statistics_version(full_experimental_set_data: str):
    experimental_set = json.load(open(full_experimental_set_data, 'r'))
    summary_statistics_version = {}

    for simulation_name, simulation_results in experimental_set.items():
        simulation_summary_stats = {}
        for variable_name, variable_results in simulation_results.items():
            hourly = np.array(variable_results["hourly"], dtype=float)
            simulation_summary_stats[variable_name] = {
                "min": float(np.min(hourly)),
                "max": float(np.max(hourly)),
                "mean": float(np.mean(hourly)),
                "std": float(np.std(hourly))
            }

        summary_statistics_version[simulation_name] = simulation_summary_stats

    out_path = Path(full_experimental_set_data).with_name("summary_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary_statistics_version, f, indent=2)


def generate(epw: str, out_root: str = "experimental_energyplus_simulations"):
    random.seed(42)
    IDF.setiddname(config.IDD_FILE_PATH)

    baseline = {
        "air_change_rate": 2.0,
        "hvac_heating_cop": 0.8,
        "hvac_cooling_cop": 3.0,
        "window_u_value": 2.0,
        "wall_r_value": 13.0,
        "roof_r_value": 30.0,
        "hvac_system_type": "gas_furnace",
    }

    levels = {
        "wall_r_value":      [4.0, 7.0, 13.0, 20.0, 30.0],
        "roof_r_value":      [10.0, 20.0, 30.0, 40.0, 50.0],
        "hvac_heating_cop":  [0.7, 0.8, 0.9, 0.95, 1.0],
        "hvac_cooling_cop":  [1.0, 2.0, 3.0, 3.5, 4.0],
    }
    keys = list(levels.keys())

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- IDF generation (625 items) + meta capture ----
    meta = {}
    combos = list(product(*(levels[k] for k in keys)))
    for i, vals in enumerate(combos, 1):
        params = baseline | dict(zip(keys, vals))
        params["air_change_rate"] = baseline["air_change_rate"] * random.uniform(0.9, 1.1)

        name = f"ff_{i:04d}"
        sim_dir = out_root / name
        idf_path = sim_dir / "in.idf"
        sim_dir.mkdir(parents=True, exist_ok=True)

        # save meta for later text generation/scoring
        meta[name] = {
            "wall_r_value": params["wall_r_value"],
            "roof_r_value": params["roof_r_value"],
            "hvac_heating_cop": params["hvac_heating_cop"],
            "hvac_cooling_cop": params["hvac_cooling_cop"],
            "air_change_rate": params["air_change_rate"],
        }

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon",
                             "coordinates": [[[0,0],[0,10],[10,10],[10,0],[0,0]]]},
                "properties": {
                    "name": name,
                    "Total Square Feet Living Area": 1000,
                    "height_ft": 10,
                    "conditioned": True,
                    **params
                }
            }]
        }
        generate_idf_from_geojson(geojson, str(idf_path))

    # write meta
    data_dir = Path("energyplus_data")
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "factorial_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- EnergyPlus runs ----
    epw = str(Path(epw))
    test_idfs = sorted(glob.glob(str(out_root / "ff_*" / "in.idf")))
    for idf in test_idfs:
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(idf))
            subprocess.run([
                "energyplus",
                "-w", os.path.abspath(epw),
                "-d", "simulation_output",
                "-r",
                "in.idf"
            ], check=True)
            print(f"Ran EnergyPlus: {idf}")
        except Exception as e:
            print(f"EnergyPlus failed at {idf}: {e}")
        finally:
            os.chdir(cwd)

    # ---- Parse & summarize ----
    experimental_json = str(data_dir / "experimental_set.json")
    _parse_eplusout_csvs_to_json(str(out_root), experimental_json)
    _generate_summary_statistics_version(experimental_json)
    print("Wrote:", experimental_json, "and summary_stats.json, and factorial_meta.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw", required=True)
    ap.add_argument("--out-root", default="experimental_energyplus_simulations")
    a = ap.parse_args()
    raise SystemExit(generate(epw=a.epw, out_root=a.out_root))