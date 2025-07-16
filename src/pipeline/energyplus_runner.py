import os
import glob
import json
import shutil
import subprocess
from pipeline.idf_generation import generate_idf_from_geojson


def run_energyplus_simulation(home_dir: str, weather_path: str) -> None:
    """
    Runs EnergyPlus simulation from within the home directory.

    Args:
        home_dir (str): Path to the directory containing 'in.idf'.
        weather_path (str): Absolute path to the weather .epw file.
    """
    original_cwd = os.getcwd()
    abs_weather_path = os.path.abspath(weather_path)

    os.makedirs(os.path.join(home_dir, "simulation_output"), exist_ok=True)
    os.chdir(home_dir)

    try:
        subprocess.run([
            "energyplus",
            "-w", abs_weather_path,
            "-d", "simulation_output",
            "-r",
            "expanded.idf"
        ], check=True)
    finally:
        os.chdir(original_cwd)


def simulate_home(home_folder_name: str) -> None:
    """
    Simulates EnergyPlus for a single home directory.

    Args:
        home_folder_name (str): Path to the home directory within the dataset.
    """
    geojson_path = os.path.join(home_folder_name, "cleaned.geojson")
    preprocessed_path = os.path.join(home_folder_name, "preprocessed.json")
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            generate_idf_from_geojson(geojson_data, os.path.join(home_folder_name, "in.idf"))

        with open(preprocessed_path, 'r') as f:
            preprocessed_data = json.load(f)
            weather_station = preprocessed_data["weather"]

        run_energyplus_simulation(
            home_folder_name,
            os.path.join('weather', f'{weather_station}.epw')
        )
    except Exception as e:
        print(f"[ERROR] Could not run EnergyPlus simulation for {home_folder_name}: {e}")
        print(f"[ERROR] This folder will be removed from the dataset.")
        shutil.rmtree(home_folder_name)
        return

    print(f"Completed simulation for {home_folder_name}.")


def simulate_all_homes(dataset_dir: str = "dataset") -> None:
    """
    Runs EnergyPlus simulations for all homes in the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory.
    """
    homes = glob.glob(os.path.join(dataset_dir, "*"))
    for home in homes:
        simulate_home(home)