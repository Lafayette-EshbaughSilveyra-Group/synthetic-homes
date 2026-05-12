"""
This script calculates the difference of generated and reported
building areas (in square feet) in the dataset.
"""

from pathlib import Path
from pyproj import Geod

import json
import numpy as np
from scipy.stats import pearsonr

if __name__ == '__main__':
    dataset_folder = Path("../dataset")

    geod = Geod(ellps='WGS84')

    actual_areas = []
    generated_areas = []

    signed_errors = []
    abs_errors = []
    percent_errors = []
    abs_percent_errors = []

    for file in dataset_folder.iterdir():
        if not file.is_dir():
            continue

        geojson_path = file / "cleaned.geojson"

        if not geojson_path.exists():
            continue

        with open(geojson_path) as f:
            geojson = json.load(f)

        try:
            actual = geojson['features'][0]['properties']['floor_area']

            geometry = geojson['features'][0]['geometry']

            ring = geometry['coordinates'][0]

            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]

            area_m2, _ = geod.polygon_area_perimeter(lons, lats)
            area_m2 = abs(area_m2)

            generated = area_m2 * 10.76391  # convert m² → ft²

            error = actual - generated
            abs_error = abs(error)

            percent_error = (error / actual) * 100
            abs_percent_error = abs(percent_error)

            actual_areas.append(actual)
            generated_areas.append(generated)

            signed_errors.append(error)
            abs_errors.append(abs_error)

            percent_errors.append(percent_error)
            abs_percent_errors.append(abs_percent_error)

        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    # Convert to numpy arrays
    actual_areas = np.array(actual_areas)
    generated_areas = np.array(generated_areas)

    signed_errors = np.array(signed_errors)
    abs_errors = np.array(abs_errors)

    percent_errors = np.array(percent_errors)
    abs_percent_errors = np.array(abs_percent_errors)

    # -----------------------------
    # Summary Statistics
    # -----------------------------

    mean_signed_error = np.mean(signed_errors)
    median_signed_error = np.median(signed_errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(signed_errors ** 2))

    mean_abs_percent_error = np.mean(abs_percent_errors)
    median_abs_percent_error = np.median(abs_percent_errors)

    # Coverage metrics
    within_10 = np.mean(abs_percent_errors <= 10) * 100
    within_20 = np.mean(abs_percent_errors <= 20) * 100
    within_30 = np.mean(abs_percent_errors <= 30) * 100

    # Outliers
    outliers_50 = np.sum(abs_percent_errors > 50)

    # Correlation
    r, p_value = pearsonr(actual_areas, generated_areas)
    r_squared = r ** 2

    # -----------------------------
    # Print Results
    # -----------------------------

    print("=" * 50)
    print("Geometry Fidelity Statistics")
    print("=" * 50)

    print(f"Number of homes: {len(actual_areas)}")

    print("\nError Metrics")
    print("-" * 50)

    print(f"Mean signed error (ft²): {mean_signed_error:.2f}")
    print(f"Median signed error (ft²): {median_signed_error:.2f}")

    print(f"MAE (ft²): {mae:.2f}")
    print(f"RMSE (ft²): {rmse:.2f}")

    print(f"Mean absolute percent error (%): {mean_abs_percent_error:.2f}")
    print(f"Median absolute percent error (%): {median_abs_percent_error:.2f}")

    print("\nCoverage")
    print("-" * 50)

    print(f"Within 10%: {within_10:.2f}%")
    print(f"Within 20%: {within_20:.2f}%")
    print(f"Within 30%: {within_30:.2f}%")

    print("\nOutliers")
    print("-" * 50)

    print(f">50% error homes: {outliers_50}")

    print("\nCorrelation")
    print("-" * 50)

    print(f"Pearson r: {r:.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"Correlation p-value: {p_value:.6f}")