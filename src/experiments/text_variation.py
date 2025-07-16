import json
import numpy as np
from pipeline.postprocessing import label_data
from experiments import HVAC_TEXT_SAMPLES, INSULATION_TEXT_SAMPLES


def run(client, runs_per_sample=5):
    """
        Objective: Determine whether variations in inspection text affect recommendations.

        Inputs:
            - multiple inspection text samples per feature (HVAC, Insulation)
            - a neutral EnergyPlus output for consistency

        Returns: results dictionary with identifiers, text samples, and mean/std output values
        """

    dataset = json.load(open('energyplus_data/experimental_set_summary_stats.json', 'r'))
    example = next(iter(dataset.values()))

    results = {
        "hvac": [],
        "insulation": [],
    }

    # HVAC text variation experiment
    for i, text in enumerate(HVAC_TEXT_SAMPLES):
        label_results = []

        for _ in range(runs_per_sample):
            result = label_data(example, text, None, client)
            label_results.append([result['insulation'], result['hvac']])

        label_result_mat = np.array(label_results)
        mean = np.mean(label_result_mat, axis=0)
        std = np.std(label_result_mat, axis=0)

        results["hvac"].append({
            "text_id": f"HVAC{i + 1}",
            "text_sample": text,
            "mean_insulation": mean[0],
            "std_insulation": std[0],
            "mean_hvac": mean[1],
            "std_hvac": std[1],
        })

    # Insulation text variation experiment
    for i, text in enumerate(INSULATION_TEXT_SAMPLES):
        label_results = []

        for _ in range(runs_per_sample):
            result = label_data(example, text, None, client)
            label_results.append([result['insulation'], result['hvac']])

        label_result_mat = np.array(label_results)
        mean = np.mean(label_result_mat, axis=0)
        std = np.std(label_result_mat, axis=0)

        results["insulation"].append({
            "text_id": f"INS{i + 1}",
            "text_sample": text,
            "mean_insulation": mean[0],
            "std_insulation": std[0],
            "mean_hvac": mean[1],
            "std_hvac": std[1],
        })

    return results
