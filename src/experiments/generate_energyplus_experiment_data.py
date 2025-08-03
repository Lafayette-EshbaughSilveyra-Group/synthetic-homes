import copy
import json
import os
import glob
import random
import subprocess
import pandas as pd
import numpy as np
from eppy.modeleditor import IDF

import config
from pipeline.idf_generation import generate_idf_from_geojson


# === Parsing Functions ===

def parse_eplusout_csvs_to_json(parent_folder: str, output_json_path: str):
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
                    "average": round(series.mean(), 3),
                    "min": round(series.min(), 3),
                    "max": round(series.max(), 3),
                    "hourly": [round(x, 3) for x in series.tolist()]
                }

        all_results[home_name] = home_data

    with open(output_json_path, "w") as f:
        json.dump(all_results, f, indent=2)


def generate_summary_statistics_version(full_experimental_set_data):
    experimental_set = json.load(open(full_experimental_set_data, 'r'))
    summary_statistics_version = {}

    for simulation_name, simulation_results in experimental_set.items():
        simulation_summary_stats = {}
        for variable_name, variable_results in simulation_results.items():
            hourly_data = np.array(variable_results["hourly"])
            simulation_summary_stats[variable_name] = {
                "min": variable_results['min'],
                "max": variable_results['max'],
                "mean": np.mean(hourly_data),
                "std": np.std(hourly_data)
            }

        summary_statistics_version[simulation_name] = simulation_summary_stats

    with open(os.path.join(os.path.dirname(full_experimental_set_data), 'summary_stats.json'),
              'w') as f:
        json.dump(summary_statistics_version, f, indent=2)


def main():
    IDF.setiddname(config.IDD_FILE_PATH)

    # === Baseline performance parameters ===
    baseline = {
        "air_change_rate": 2.0,
        "hvac_heating_cop": 0.8,  # Gas furnace efficiency (e.g. 0.8 = 80%)
        "hvac_cooling_cop": 3.0,  # Cooling COP
        "window_u_value": 2.0,
        "wall_r_value": 13.0,
        "roof_r_value": 30.0,
        "hvac_system_type": "gas_furnace"
    }

    # === Variation definitions ===
    variations = {
        "wall_r_value": [4.0, 7.0, 13.0, 20.0, 30.0],  # Higher => better insulation
        "roof_r_value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "hvac_heating_cop": [0.7, 0.8, 0.9, 0.95, 1.0],  # Gas furnace efficiency <= 1.0
        "hvac_cooling_cop": [1.0, 2.0, 3.0, 3.5, 4.0],  # Cooling COP values
        # "air_change_rate_variations": [
        #     0.75,  # Leaky pre-1980s home
        #     0.5,  # Moderately leaky older home
        #     0.35,  # Typical code-minimum home
        #     0.2,  # Tight new construction
        #     0.1  # Very tight house (Passive House level)
        # ],
    }

    # === Generation Loop ===
    for param, values in variations.items():
        for i, val in enumerate(values):
            test_params = copy.deepcopy(baseline)
            test_params[param] = val

            # Add some variation into the ACHR
            test_params['air_change_rate'] = baseline['air_change_rate'] * random.uniform(0.9, 1.1)

            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]
                        },
                        "properties": {
                            "name": f"test_{param}_{i}",
                            "Total Square Feet Living Area": 1000,
                            "height_ft": 10,
                            "conditioned": True,
                            **test_params
                        }
                    }
                ]
            }

            # Generate IDF
            idf_path = f"experimental_energyplus_simulations/test_{param}_{i + 1}/in.idf"
            os.makedirs(os.path.dirname(idf_path), exist_ok=True)
            generate_idf_from_geojson(geojson, idf_path)

    # === Simulation Loop ===

    test_idfs = glob.glob("experimental_energyplus_simulations/test_*/in.idf")

    for idf in test_idfs:
        abs_weather_path = os.path.abspath('weather/KMSP.epw')
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(idf))

            subprocess.run([
                "energyplus",
                "-w", abs_weather_path,
                "-d", "simulation_output",
                "-r",
                'expanded.idf'
            ], check=True)

            print(f"✅ Successfully ran EnergyPlus on {idf}")
        except Exception as e:
            print(f"❌ Failed to run EnergyPlus: {e}")
        finally:
            os.chdir(cwd)

    # === Execute parsing and summarizing ===

    os.makedirs(os.path.join(os.getcwd(), 'energyplus_data'), exist_ok=True)

    parse_eplusout_csvs_to_json(os.getcwd(), os.path.join(os.getcwd(), 'energyplus_data', "experimental_set.json"))
    generate_summary_statistics_version(os.path.join(os.getcwd(), 'energyplus_data', "experimental_set.json"))
