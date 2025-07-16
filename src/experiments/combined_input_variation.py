import json
import numpy as np
from pipeline.postprocessing import label_data
from experiments import HVAC_TEXT_SAMPLES, INSULATION_TEXT_SAMPLES


def run(client, runs_per_sample=5):
    """
    Objective: Determine interaction effects between inspection text and EnergyPlus outputs.

    Inputs:
        - Extreme inspection texts (very bad vs. very good)
        - Extreme EnergyPlus outputs (very bad vs. very good)

    Returns:
        Dictionary summarizing mean/std results for each experimental condition.
    """

    dataset = json.load(open('energyplus_data/experimental_set_summary_stats.json', 'r'))

    experimental_groups = [
        [
            (HVAC_TEXT_SAMPLES[4], dataset['test_hvac_cooling_cop_5']),
            (HVAC_TEXT_SAMPLES[4], dataset['test_hvac_cooling_cop_1']),
            (HVAC_TEXT_SAMPLES[0], dataset['test_hvac_cooling_cop_5']),
            (HVAC_TEXT_SAMPLES[0], dataset['test_hvac_cooling_cop_1']),
        ],
        [
            (HVAC_TEXT_SAMPLES[4], dataset['test_hvac_heating_cop_5']),
            (HVAC_TEXT_SAMPLES[4], dataset['test_hvac_heating_cop_1']),
            (HVAC_TEXT_SAMPLES[0], dataset['test_hvac_heating_cop_5']),
            (HVAC_TEXT_SAMPLES[0], dataset['test_hvac_heating_cop_1']),
        ],
        [
            (INSULATION_TEXT_SAMPLES[4], dataset['test_roof_r_value_5']),
            (INSULATION_TEXT_SAMPLES[4], dataset['test_roof_r_value_1']),
            (INSULATION_TEXT_SAMPLES[0], dataset['test_roof_r_value_5']),
            (INSULATION_TEXT_SAMPLES[0], dataset['test_roof_r_value_1'])
        ],
        [
            (INSULATION_TEXT_SAMPLES[4], dataset['test_wall_r_value_5']),
            (INSULATION_TEXT_SAMPLES[4], dataset['test_wall_r_value_1']),
            (INSULATION_TEXT_SAMPLES[0], dataset['test_wall_r_value_5']),
            (INSULATION_TEXT_SAMPLES[0], dataset['test_wall_r_value_1'])
        ],
    ]

    results = {}

    for eg_idx, eg in enumerate(experimental_groups):
        eg_results = {}

        for element_idx, (text, eplus) in enumerate(eg):
            label_results = [
                label_data(eplus, text, None, client)
                for _ in range(runs_per_sample)
            ]

            label_result_mat = np.array([
                [res['insulation'], res['hvac']]
                for res in label_results
            ])

            mean = np.mean(label_result_mat, axis=0)
            std = np.std(label_result_mat, axis=0)

            name_map = {
                0: "good_text_good_simulation",
                1: "good_text_poor_simulation",
                2: "poor_text_good_simulation",
                3: "poor_text_poor_simulation",
            }

            eg_results[name_map[element_idx]] = {
                "mean_insulation": mean[0],
                "std_insulation": std[0],
                "mean_hvac": mean[1],
                "std_hvac": std[1],
            }

        group_name_map = {
            0: "hvac_cooling",
            1: "hvac_heating",
            2: "insulation_roof_r",
            3: "insulation_wall_r",
        }

        results[group_name_map[eg_idx]] = eg_results

    return results
