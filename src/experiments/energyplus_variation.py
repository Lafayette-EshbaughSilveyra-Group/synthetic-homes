import json
import numpy as np
from pipeline.postprocessing import label_data


# Helper functions to adapt summary_stats.json entry to calibrated labeler input
def _resolve_key(entry: dict, wanted: str) -> str | None:
    """
    Return the key in `entry` that matches `wanted` allowing for prefixes, case and trailing spaces.
    Matching order:
      1) exact
      2) stripped exact
      3) case-insensitive exact
      4) suffix match on stripped keys (handles prefixed keys like "FF_0001 ... Electricity:HVAC [J](Hourly) ")
    """
    if wanted in entry:
        return wanted
    stripped = {k.strip(): k for k in entry.keys()}
    lower = {k.strip().lower(): k for k in entry.keys()}
    w = wanted.strip()
    if w in stripped:
        return stripped[w]
    if w.lower() in lower:
        return lower[w.lower()]
    # suffix match
    candidates = [k for k in entry.keys() if k.strip().endswith(w)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates, key=len)[0]
    return None

def _summary_to_labeler_features(summary_entry: dict) -> dict:
    """
    Build the minimal `results_json` structure expected by the calibrated labeler from a single
    `summary_stats.json` entry. Only `hvac_electricity` and `heating_coil` averages are required.
    """
    hvac_key = _resolve_key(summary_entry, "Electricity:HVAC [J](Hourly)")
    heat_key = _resolve_key(summary_entry, "Heating Coil Heating Energy [J](Hourly)")
    if hvac_key is None or heat_key is None:
        keys_preview = list(summary_entry.keys())[:8]
        raise KeyError(f"Could not resolve required keys from summary entry. Found keys (sample): {keys_preview}")
    hvac_avg = float(summary_entry[hvac_key]["mean"])
    heat_avg = float(summary_entry[heat_key]["mean"])
    return {
        "zone": "FACTORIAL_HOME",
        "features": {
            "hvac_electricity": {"average": hvac_avg, "min": hvac_avg, "max": hvac_avg, "hourly": []},
            "heating_coil":     {"average": heat_avg, "min": heat_avg, "max": heat_avg, "hourly": []},
        }
    }


def run(client, runs_per_sample=5):
    """
    Objective: Determine whether EnergyPlus output differences affect retrofit recommendations.

    Inputs:
        - Neutral inspection text
        - EnergyPlus outputs from energyplus_data/experimental_set_summary_stats.json

    Returns:
        List of dictionaries with example IDs and per-feature mean/std values.
    """
    neutral_text = (
        "The home has two stories with vinyl siding and a shingled roof. "
        "Windows appear to be single-hung with no visible damage. "
        "The HVAC system is located on the first floor near the utility room. "
        "Insulation levels in the attic are unknown. "
        "Doors are wood-core with standard weather stripping."
    )

    from pathlib import Path
    dataset_path = Path(__file__).resolve().parents[2] / "energyplus_data" / "summary_stats.json"
    dataset = json.load(open(dataset_path, 'r'))

    example_name_map = {
        "test_hvac_cooling_cop_": "HVACC",
        "test_hvac_heating_cop_": "HVACH",
        "test_roof_r_value_": "ROOFR",
        "test_wall_r_value_": "WALLR",
    }

    results = []

    for example_name, example in dataset.items():
        results_json = _summary_to_labeler_features(example)
        label_results = [
            label_data(
                results_json, neutral_text, None, client,
                use_calibrated_fusion=True,
                energyplus_label_method="raw",
                hvac_scaler_path="energyplus_data/hvac_scaler_params.json",
                ins_scaler_path="energyplus_data/insulation_scaler_params.json",
                tau=0.0
            )
            for _ in range(runs_per_sample)
        ]

        label_result_mat = np.array([
            [res['insulation'], res['hvac']] for res in label_results
        ])

        mean = np.mean(label_result_mat, axis=0)
        std = np.std(label_result_mat, axis=0)

        abbrev, idx = "UNK", "0"
        for prefix, short in example_name_map.items():
            if example_name.startswith(prefix):
                abbrev = short
                idx = example_name[len(prefix):]
                break

        results.append({
            "example_id": f"{abbrev}-{idx}",
            "mean_insulation": mean[0],
            "std_insulation": std[0],
            "mean_hvac": mean[1],
            "std_hvac": std[1],
        })

    return results
