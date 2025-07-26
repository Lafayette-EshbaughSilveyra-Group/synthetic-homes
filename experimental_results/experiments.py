import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from main import label_data

RUNS_PER_SAMPLE = 5  # number of repeated runs per text sample


def exp_1_text_variation():
    """
    Objective: Determine whether variations in inspection text affect recommendations.

    Inputs:
        - multiple inspection text samples per feature (HVAC, Insulation)
        - a neutral EnergyPlus output for consistency

    Returns: results dictionary with identifiers, text samples, and mean/std output values
    """

    hvac_text_samples = [
        # Very bad (very inefficient, needs replacement)
        "There is an older HVAC unit installed, with signs of rust on the exterior.",
        # Bad (inefficient, functional but outdated)
        "The HVAC system appears to be in working condition but is an older standard-efficiency model.",
        # Medium (moderate efficiency)
        "The home uses window AC units rather than a central HVAC system.",
        # Good (efficient)
        "The HVAC system was recently replaced with a standard high-efficiency model and is expected to operate efficiently.",
        # Very good (very efficient, top-tier system)
        "A state-of-the-art HVAC system with smart thermostats and variable-speed compressors was recently installed, maximizing energy efficiency."
    ]

    insulation_text_samples = [
        # Very bad (very inefficient, minimal insulation)
        "Attic insulation is minimal, with exposed joists visible throughout, causing significant heat loss.",
        # Bad (poor insulation)
        "No signs of added insulation were observed in the basement ceiling, suggesting potential energy inefficiency.",
        # Medium (moderate insulation)
        "Walls appear to be adequately insulated based on construction year, though no upgrades were observed.",
        # Good (efficient insulation)
        "Blown-in insulation is present in the attic to a depth of approximately 10 inches, providing good thermal resistance.",
        # Very good (very efficient, top-tier insulation)
        "High-performance spray foam insulation was installed throughout the walls, attic, and basement, providing maximum energy efficiency."
    ]

    # air_change_text_samples = [
    #     # Very bad (very leaky)
    #     "Large gaps around windows and doors were observed, with noticeable drafts entering the living room.",
    #     # Bad (leaky)
    #     "Basement rim joists are unsealed, allowing air infiltration that could reduce heating efficiency.",
    #     # Medium (moderate)
    #     "No major air leaks were observed, but insulation around pipes is minimal.",
    #     # Good (tight)
    #     "Windows and doors are well-sealed with caulking and weather stripping in good condition.",
    #     # Very good (very tight)
    #     "A blower door test confirmed airtight construction exceeding Passive House standards."
    # ]

    dataset = json.load(open('energyplus_data/experimental_set_summary_stats.json', 'r'))
    example = next(iter(dataset.values()))

    results = {
        "hvac": [],
        "insulation": [],
        # "sealing": []
    }

    # HVAC text variation experiment
    for i, text in enumerate(hvac_text_samples):
        label_results = []

        for _ in range(RUNS_PER_SAMPLE):
            result = label_data(example, text, None, client)
            label_results.append([result['insulation'], result['hvac'],
                                  # result['sealing']
                                  ])

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
            # "mean_sealing": mean[2],
            # "std_sealing": std[2],
        })

    # Insulation text variation experiment
    for i, text in enumerate(insulation_text_samples):
        label_results = []

        for _ in range(RUNS_PER_SAMPLE):
            result = label_data(example, text, None, client)
            label_results.append([result['insulation'], result['hvac'],
                                  # result['sealing']
                                  ])

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
            # "mean_sealing": mean[2],
            # "std_sealing": std[2],
        })

    # Air Change Rate (Infiltration) text variation experiment
    # for i, text in enumerate(air_change_text_samples):
    #     label_results = []
    #
    #     for _ in range(RUNS_PER_SAMPLE):
    #         result = label_data(example, text, None, client)
    #         label_results.append([result['insulation'], result['hvac'], result['sealing']])
    #
    #     label_result_mat = np.array(label_results)
    #     mean = np.mean(label_result_mat, axis=0)
    #     std = np.std(label_result_mat, axis=0)
    #
    #     results["sealing"].append({
    #         "text_id": f"ACR{i + 1}",
    #         "text_sample": text,
    #         "mean_insulation": mean[0],
    #         "std_insulation": std[0],
    #         "mean_hvac": mean[1],
    #         "std_hvac": std[1],
    #         "mean_sealing": mean[2],
    #         "std_sealing": std[2]
    #     })

    return results


def exp_2_energyplus_variation():
    """
    Objective: Determine whether E+ output differences affect recommendations.

    Inputs:
        - neutral inspection text
        - all E+ outputs generated in energyplus_data/experimental_set_summary_stats.json

    Returns: list of dictionaries with example IDs and per-feature mean/std values.
    """

    neutral_text = "The home has two stories with vinyl siding and a shingled roof. Windows appear to be single-hung with no visible damage. The HVAC system is located on the first floor near the utility room. Insulation levels in the attic are unknown. Doors are wood-core with standard weather stripping."

    results = []

    dataset = json.load(open('energyplus_data/experimental_set_summary_stats.json', 'r'))

    for example_name, example in dataset.items():
        label_results = []
        for _ in range(RUNS_PER_SAMPLE):
            result = label_data(example, neutral_text, None, client)
            label_results.append([result['insulation'], result['hvac'],
                                  # result['sealing']
                                  ])

        label_result_mat = np.array(label_results)
        mean = np.mean(label_result_mat, axis=0)
        std = np.std(label_result_mat, axis=0)

        example_name_map = {
            # "test_air_change_rate_variations_": "ACR",
            "test_hvac_cooling_cop_": "HVACC",
            "test_hvac_heating_cop_": "HVACH",
            "test_roof_r_value_": "ROOFR",
            "test_wall_r_value_": "WALLR",
        }

        for prefix, abbrev in example_name_map.items():
            if example_name.startswith(prefix):
                idx = example_name[len(prefix):]
                break
        else:
            abbrev = "UNK"
            idx = "0"

        results.append({
            "example_id": f"{abbrev}-{idx}",
            "mean_insulation": mean[0],
            "std_insulation": std[0],
            "mean_hvac": mean[1],
            "std_hvac": std[1],
            # "mean_sealing": mean[2],
            # "std_sealing": std[2]
        })

    return results


def exp_3_combined_input_variation():
    """
    Objective: Determine interaction effects between inspection text and EnergyPlus outputs.

    Inputs:
        - Extreme inspection texts (very bad vs. very good)
        - Extreme EnergyPlus outputs (very bad vs. very good)

    Returns: dictionary summarizing mean/std results for each experimental condition.
    """

    hvac_text_samples = [
        # Very bad (very inefficient, needs replacement)
        "There is an older HVAC unit installed, with signs of rust on the exterior.",
        # Bad (inefficient, functional but outdated)
        "The HVAC system appears to be in working condition but is an older standard-efficiency model.",
        # Medium (moderate efficiency)
        "The home uses window AC units rather than a central HVAC system.",
        # Good (efficient)
        "The HVAC system was recently replaced with a standard high-efficiency model and is expected to operate efficiently.",
        # Very good (very efficient, top-tier system)
        "A state-of-the-art HVAC system with smart thermostats and variable-speed compressors was recently installed, maximizing energy efficiency."
    ]

    insulation_text_samples = [
        # Very bad (very inefficient, minimal insulation)
        "Attic insulation is minimal, with exposed joists visible throughout, causing significant heat loss.",
        # Bad (poor insulation)
        "No signs of added insulation were observed in the basement ceiling, suggesting potential energy inefficiency.",
        # Medium (moderate insulation)
        "Walls appear to be adequately insulated based on construction year, though no upgrades were observed.",
        # Good (efficient insulation)
        "Blown-in insulation is present in the attic to a depth of approximately 10 inches, providing good thermal resistance.",
        # Very good (very efficient, top-tier insulation)
        "High-performance spray foam insulation was installed throughout the walls, attic, and basement, providing maximum energy efficiency."
    ]

    # air_change_text_samples = [
    #     # Very bad (very leaky)
    #     "Large gaps around windows and doors were observed, with noticeable drafts entering the living room.",
    #     # Bad (leaky)
    #     "Basement rim joists are unsealed, allowing air infiltration that could reduce heating efficiency.",
    #     # Medium (moderate)
    #     "No major air leaks were observed, but insulation around pipes is minimal.",
    #     # Good (tight)
    #     "Windows and doors are well-sealed with caulking and weather stripping in good condition.",
    #     # Very good (very tight)
    #     "A blower door test confirmed airtight construction exceeding Passive House standards."
    # ]

    dataset = json.load(open('energyplus_data/experimental_set_summary_stats.json', 'r'))

    experimental_groups = [
        [
            (hvac_text_samples[4], dataset['test_hvac_cooling_cop_5']),
            (hvac_text_samples[4], dataset['test_hvac_cooling_cop_1']),
            (hvac_text_samples[0], dataset['test_hvac_cooling_cop_5']),
            (hvac_text_samples[0], dataset['test_hvac_cooling_cop_1']),
        ],
        [
            (hvac_text_samples[4], dataset['test_hvac_heating_cop_5']),
            (hvac_text_samples[4], dataset['test_hvac_heating_cop_1']),
            (hvac_text_samples[0], dataset['test_hvac_heating_cop_5']),
            (hvac_text_samples[0], dataset['test_hvac_heating_cop_1']),
        ],
        [
            (insulation_text_samples[4], dataset['test_roof_r_value_5']),
            (insulation_text_samples[4], dataset['test_roof_r_value_1']),
            (insulation_text_samples[0], dataset['test_roof_r_value_5']),
            (insulation_text_samples[0], dataset['test_roof_r_value_1'])
        ],
        [
            (insulation_text_samples[4], dataset['test_wall_r_value_5']),
            (insulation_text_samples[4], dataset['test_wall_r_value_1']),
            (insulation_text_samples[0], dataset['test_wall_r_value_5']),
            (insulation_text_samples[0], dataset['test_wall_r_value_1'])
        ],
        # [
        #     (air_change_text_samples[4], dataset['test_air_change_rate_variations_5']),  # good text + good sim
        #     (air_change_text_samples[4], dataset['test_air_change_rate_variations_1']),  # good text + poor sim
        #     (air_change_text_samples[0], dataset['test_air_change_rate_variations_5']),  # poor text + good sim
        #     (air_change_text_samples[0], dataset['test_air_change_rate_variations_1']),  # poor text + poor sim
        # ]
    ]

    results = {}

    for eg_idx, eg in enumerate(experimental_groups):

        eg_results = {}

        for element_idx, (text, eplus) in enumerate(eg):

            label_results = []

            for _ in range(RUNS_PER_SAMPLE):
                result = label_data(eplus, text, None, client)
                label_results.append([result['insulation'], result['hvac'],
                                      # result['sealing']
                                      ])

            label_result_mat = np.array(label_results)
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
                # "mean_sealing": mean[2],
                # "std_sealing": std[2],
            }

        name_map = {
            0: "hvac_cooling",
            1: "hvac_heating",
            2: "insulation_roof_r",
            3: "insulation_wall_r",
            # 4: "sealing_air_change_rate"
        }

        results[name_map[eg_idx]] = eg_results

    return results


def save_results(results, experiment_name):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)


EXPERIMENTS = {
    1: exp_1_text_variation,
    2: exp_2_energyplus_variation,
    3: exp_3_combined_input_variation
}

EXPERIMENT_NAMES = {
    1: "text_variation",
    2: "energyplus_variation",
    3: "combined_input_variation"
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: experiments.py <experiment_id> [experiment_id] ...')
        exit(1)

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)

    experiment_ids = sys.argv[1:]

    if experiment_ids[0] == 'all':
        for exp_id, exp in EXPERIMENTS.items():
            results = exp()
            save_results(results, EXPERIMENT_NAMES[exp_id])
        exit(0)

    for experiment_id in experiment_ids:
        results = EXPERIMENTS[int(experiment_id)]()
        save_results(results, EXPERIMENT_NAMES[int(experiment_id)])
