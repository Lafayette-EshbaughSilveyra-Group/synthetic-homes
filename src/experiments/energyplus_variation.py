import json
import numpy as np
from pipeline.postprocessing import label_data


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

    dataset_path = 'energyplus_data/experimental_set_summary_stats.json'
    dataset = json.load(open(dataset_path, 'r'))

    example_name_map = {
        "test_hvac_cooling_cop_": "HVACC",
        "test_hvac_heating_cop_": "HVACH",
        "test_roof_r_value_": "ROOFR",
        "test_wall_r_value_": "WALLR",
    }

    results = []

    for example_name, example in dataset.items():
        label_results = [
            label_data(example, neutral_text, None, client)
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
