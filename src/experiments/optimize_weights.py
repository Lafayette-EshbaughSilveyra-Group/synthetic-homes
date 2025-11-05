import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import csv

from experiments.constants import HVAC_TEXT_SAMPLES
from pipeline.postprocessing import label_data
import matplotlib.pyplot as plt


def absolute_difference_of_experimental_outcomes(good_text_bad_sim: float, bad_text_good_sim: float) -> float:
    return abs(good_text_bad_sim - bad_text_good_sim)


def run(client, average_of: int = 5) -> Dict[str, Any]:
    """
    Sweep text/sim weights (1..99 vs 99..1), compute the absolute difference of experimental outcomes
    for (good text + bad sim) vs (bad text + good sim), and return the best weights and all results.

    Returns:
        {
          "best": {"weights": (text_weight, sim_weight), "diff": float},
          "results": [((text_weight, sim_weight), diff), ...]
        }
    """
    # Inputs
    bad_text = HVAC_TEXT_SAMPLES[0]
    good_text = HVAC_TEXT_SAMPLES[4]

    dataset_path = Path(__file__).resolve().parents[2] / "energyplus_data" / "summary_stats.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    bad_sim = dataset["test_hvac_cooling_cop_1"]
    good_sim = dataset["test_hvac_cooling_cop_5"]

    results: List[Tuple[Tuple[int, int], float]] = []

    for i in range(1, 100):  # 1..99 inclusive
        text_weight = i
        sim_weight = 100 - i

        total = 0.0
        for _ in range(average_of):
            out = label_data(
                bad_sim, good_text, home_dir_name=None, client=client,
                text_weight=text_weight, energyplus_weight=sim_weight
            )
            total += out["hvac"]
        avg_good_text_bad_sim = total / average_of

        total = 0.0
        for _ in range(average_of):
            out = label_data(
                good_sim, bad_text, home_dir_name=None, client=client,
                text_weight=text_weight, energyplus_weight=sim_weight
            )
            total += out["hvac"]
        avg_bad_text_good_sim = total / average_of

        diff = absolute_difference_of_experimental_outcomes(
            avg_good_text_bad_sim, avg_bad_text_good_sim
        )

        results.append(((text_weight, sim_weight), diff))

    # Find best weights (min diff)
    best_weights, best_diff = min(results, key=lambda x: x[1])

    # Print a summary
    print(f"[Different Weights Experiment] Best weights: text={best_weights[0]}%, sim={best_weights[1]}% (diff={best_diff:.6f})")

    return {"best": {"weights": best_weights, "diff": best_diff}, "results": results}