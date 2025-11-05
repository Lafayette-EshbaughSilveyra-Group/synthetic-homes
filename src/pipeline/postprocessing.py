import os
import glob
import json

import numpy as np
import pandas as pd
import re
import time
from openai import RateLimitError
from typing import Any

import config


def extract_results_from_csv(home_dir: str) -> dict:
    """
    Extracts summary statistics and hourly zone temperatures from an EnergyPlus CSV output.

    Parameters:
        home_dir (str): Name of the home (i.e., RAMBEAU_RD_15)

    Returns:
        dict: Structured dictionary of zone-level features.
    """
    df = pd.read_csv(f"{home_dir}/simulation_output/eplusout.csv")
    df.columns = df.columns.str.strip()

    air_col = "GENERATED_HOME:Zone Air Temperature [C](Hourly)"
    heating_coil = "GENERATED_HOME PTAC HEATING COIL:Heating Coil Heating Energy [J](Hourly)"
    facility_electricity = "Electricity:Facility [J](Hourly)"
    hvac_electricity = "Electricity:HVAC [J](Hourly)"

    def compute_stats(series):
        return {
            "average": round(series.mean(), 3),
            "min": round(series.min(), 3),
            "max": round(series.max(), 3),
            "hourly": [round(x, 3) for x in series.tolist()]
        }

    return {
        "zone": "GENERATED_HOME",
        "features": {
            "air_temperature": compute_stats(df[air_col]),
            "heating_coil": compute_stats(df[heating_coil]),
            "facility_electricity": compute_stats(df[facility_electricity]),
            "hvac_electricity": compute_stats(df[hvac_electricity]),
        }
    }


# src/pipeline/fusion/fuse_equal.py
import json
from pathlib import Path

_cache = {}


def _load_params(path: str | Path):
    path = str(path)
    if path not in _cache:
        _cache[path] = json.load(open(path))
    return _cache[path]


def fuse_equal_from_raw(text_raw: float, sim_raw: float, *,
                        scaler_path="energyplus_data/hvac_scaler_params.json",
                        tau: float = 0.0):
    params = _load_params(scaler_path)
    muT, sdT = params["text"]["mean"], params["text"]["std"]
    muS, sdS = params["sim"]["mean"], params["sim"]["std"]

    zt = (text_raw - muT) / (sdT or 1e-8)
    zs = (sim_raw - muS) / (sdS or 1e-8)
    fused = 0.5 * zt + 0.5 * zs

    # optional symmetric abstain band
    decision = 1 if fused > tau else (0 if fused < -tau else -1)
    return {"fused": fused, "decision": decision, "z_text": zt, "z_sim": zs}


def _get_sim_raw_for_concept(results: dict, concept: str) -> float:
    """
    Returns a RAW simulation scalar (no 0-1 normalization) for the given concept.
    This should match what you used when building the scaler.

    For HVAC we take Electricity:HVAC mean J/hour.
    For Insulation we take Heating Coil Heating Energy mean J/hour (proxy for envelope load).

    Adjust if your scaler was fit on different variables.
    """
    # keys per your extract_results_from_csv()
    # "hvac_electricity": "Electricity:HVAC [J](Hourly)"
    # "heating_coil":     "Heating Coil Heating Energy [J](Hourly)"
    if concept == "hvac":
        return float(results["features"]["hvac_electricity"]["average"])
    elif concept == "insulation":
        return float(results["features"]["heating_coil"]["average"])
    else:
        raise ValueError(f"Unknown concept: {concept}")


def label_data(results_json: dict, inspection_report: str, home_dir_name: str, client: Any,
               text_weight: float = 0.2, energyplus_weight: float = 0.8,
               energyplus_label_method: str = 'heuristic',
               use_calibrated_fusion: bool = True,
               hvac_scaler_path: str = "energyplus_data/hvac_scaler_params.json",
               ins_scaler_path: str = "energyplus_data/insulation_scaler_params.json",
               tau: float = 0.0) -> dict[str, float | Any]:
    """
    Uses OpenAI API to label a datapoint based on its results.json and inspection report.
    If `use_calibrated_fusion` is True and scaler files exist, we fuse per concept via z-scored 50/50.
    Otherwise we fall back to the legacy weighted average.
    """

    def build_text_prompt(inspection_report: str) -> str:
        return f"""
You are an expert building energy analyst.

Below is a **narrative inspection report** for a building.

Your task is to assign a **confidence score** in the range [0, 1] for the **need** for each of the following retrofits:
- Insulation upgrade
- HVAC upgrade

### IMPORTANT:
- A value of 0 means \"definitely not needed\".
- A value of 1 means \"definitely needed\".
- Intermediate values (e.g. 0.33, 0.5, 0.75) indicate uncertainty or partial need.

### INSPECTION REPORT (free text):
\"\"\"
{inspection_report}
\"\"\"

### RESPONSE FORMAT:
Return a JSON object like:
{{
  \"insulation\": 0.5,
  \"hvac\": 0.5,
}}

Only include the JSON. No explanation or commentary.
"""

    def build_energyplus_prompt(results_json: dict) -> str:
        return f"""
You are an expert building energy analyst.

Below are **EnergyPlus simulation results** for a building.

Your task is to assign a **confidence score** in the range [0, 1] for the **need** for each of the following retrofits:
- Insulation upgrade
- HVAC upgrade

### IMPORTANT:
- A value of 0 means \"definitely not needed\".
- A value of 1 means \"definitely needed\".
- Intermediate values (e.g. 0.33, 0.5, 0.75) indicate uncertainty or partial need.

### ENERGYPLUS SIMULATION RESULTS (json):

{json.dumps(results_json, indent=2)}

### RESPONSE FORMAT:
Return a JSON object like:
{{
  \"insulation\": 0.5,
  \"hvac\": 0.5,
}}

Only include the JSON. No explanation or commentary.
"""

    def safe_chat_response(prompt: str):
        while True:
            try:
                return client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
            except RateLimitError:
                print("Rate limit hit — sleeping for 2 seconds...")
                time.sleep(2)

    def extract_json_from_response(response_text: str):
        cleaned = re.sub(r"^```(\w+)?", "", response_text.strip())
        cleaned = re.sub(r"```$", "", cleaned.strip())
        return cleaned

    def heuristic_labeler(results: dict):
        # Dynamically compute best and worst values from all homes in dataset
        dataset_dir = "dataset"
        heating_loads = []
        hvac_loads = []

        homes = glob.glob(os.path.join(dataset_dir, "*"))
        for home in homes:
            results_path = os.path.join(home, "results.json")
            if os.path.isfile(results_path):
                with open(results_path, "r") as f:
                    res = json.load(f)
                heating_load_hourly_J = 0.0
                hvac_hourly_J = 0.0
                for var_name, var_data in res.items():
                    if "Heating Coil Heating Energy" in var_name:
                        heating_load_hourly_J = float(var_data.get("average", var_data.get("mean", 0)))
                    elif "Electricity:HVAC" in var_name:
                        hvac_hourly_J = float(var_data.get("average", var_data.get("mean", 0)))
                heating_load_annual_kWh = (heating_load_hourly_J * 730 * 12) / 3600000
                hvac_annual_kWh = (hvac_hourly_J * 730 * 12) / 3600000
                heating_loads.append(heating_load_annual_kWh)
                hvac_loads.append(hvac_annual_kWh)

        if heating_loads:
            hl_best = min(heating_loads)
            hl_worst = max(heating_loads)
        else:
            hl_best = 0
            hl_worst = 1
        if hvac_loads:
            hvac_best = min(hvac_loads)
            hvac_worst = max(hvac_loads)
        else:
            hvac_best = 0
            hvac_worst = 1

        heating_load_hourly_J = 0.0
        hvac_hourly_J = 0.0

        for var_name, var_data in results.items():
            if "Heating Coil Heating Energy" in var_name:
                heating_load_hourly_J = float(var_data["mean"])
            elif "Electricity:HVAC" in var_name:
                hvac_hourly_J = float(var_data["mean"])

        heating_load_annual_kWh = (heating_load_hourly_J * 730 * 12) / 3600000
        hvac_annual_kWh = (hvac_hourly_J * 730 * 12) / 3600000

        insulation_score = (heating_load_annual_kWh - hl_best) / (hl_worst - hl_best) if hl_worst != hl_best else 0
        hvac_score = (hvac_annual_kWh - hvac_best) / (hvac_worst - hvac_best) if hvac_worst != hvac_best else 0

        insulation_score = max(min(insulation_score, 1), 0)
        hvac_score = max(min(hvac_score, 1), 0)

        return {
            "insulation": insulation_score,
            "hvac": hvac_score,
        }

    try:
        text_prompt = build_text_prompt(inspection_report)
        energyplus_prompt = build_energyplus_prompt(results_json)

        text_response = safe_chat_response(text_prompt)
        text_content = extract_json_from_response(text_response.choices[0].message.content)
        text_data = json.loads(text_content)  # {"insulation": x, "hvac": y} in [0,1]

        # EnergyPlus pathway (only used for fallback path or if you explicitly want LLM 0-1)
        if energyplus_label_method == 'gpt':
            energyplus_response = safe_chat_response(energyplus_prompt)
            energyplus_content = extract_json_from_response(energyplus_response.choices[0].message.content)
            energyplus_data = json.loads(energyplus_content)  # {"insulation": x, "hvac": y} in [0,1]
        elif energyplus_label_method == 'heuristic':
            energyplus_data = heuristic_labeler(results_json)  # {"insulation": x, "hvac": y} in [0,1]
        elif energyplus_label_method == 'raw':
            # Not used directly; raw values are taken below per concept for calibrated fusion
            energyplus_data = None
        else:
            raise ValueError(f"Unsupported energyplus label method: {energyplus_label_method}")

        hvac_scaler_ok = Path(hvac_scaler_path).exists()
        ins_scaler_ok = Path(ins_scaler_path).exists()
        can_use_calibrated = use_calibrated_fusion and hvac_scaler_ok and ins_scaler_ok

        if can_use_calibrated:
            # ---- Per-concept calibrated fusion (recommended) ----
            # TEXT raws are the LLM outputs for each concept (0..1).
            text_hvac_raw = float(text_data["hvac"])
            text_ins_raw = float(text_data["insulation"])

            # SIM raws must be the *raw scalars* used in calibration (not 0..1)
            sim_hvac_raw = _get_sim_raw_for_concept(results_json, "hvac")
            sim_ins_raw = _get_sim_raw_for_concept(results_json, "insulation")

            hvac_fused = fuse_equal_from_raw(text_hvac_raw, sim_hvac_raw,
                                             scaler_path=hvac_scaler_path, tau=tau)
            ins_fused = fuse_equal_from_raw(text_ins_raw, sim_ins_raw,
                                            scaler_path=ins_scaler_path, tau=tau)

            def _sigmoid(x): return 1 / (1 + np.exp(-x))

            result = {
                "insulation": _sigmoid(float(np.clip(ins_fused["fused"], -3, 3))),  # keep a bounded range if you want
                "hvac": _sigmoid(float(np.clip(hvac_fused["fused"], -3, 3))),
                # optional debug:
                "_debug": {
                    "hvac": hvac_fused,
                    "insulation": ins_fused
                }
            }
        else:
            # ---- Fallback: legacy weighted average on 0..1 scores ----
            if energyplus_data is None:
                # If user set energyplus_label_method='raw' but we can't fuse, approximate with heuristic
                energyplus_data = heuristic_labeler(results_json)

            total_weight = text_weight + energyplus_weight
            result = {
                "insulation": (text_data["insulation"] * text_weight +
                               energyplus_data["insulation"] * energyplus_weight) / total_weight,
                "hvac": (text_data["hvac"] * text_weight +
                         energyplus_data["hvac"] * energyplus_weight) / total_weight,
                "_fallback": True,
                "_note": "Calibrated fusion unavailable; used legacy weighted average."
            }

        if home_dir_name is None:
            return result
        label_path = f"{home_dir_name}/label.json"
        with open(label_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except json.JSONDecodeError as e:
        print("Failed to parse JSON. Raw content:")
        raise e


def process_results(home_dir_name: str, client: Any) -> None:
    """
    Processes EnergyPlus results and applies labeling for a single home.

    Args:
        home_dir_name (str): Path to the home directory.
        client (Any): API client for LLM calls.
    """
    results_json = extract_results_from_csv(home_dir_name)
    json.dump(results_json, open(f'{home_dir_name}/results.json', 'w'))

    inspection_note = json.load(open(f'{home_dir_name}/cleaned.geojson', 'r'))["features"][0]["properties"][
        "inspection_note"]

    label_data(results_json, inspection_note, home_dir_name, client)


def run_postprocessing_for_dataset(dataset_dir: str = "dataset", client: Any = None) -> None:
    """
    Runs postprocessing for all homes in the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory.
        client (Any): API client for LLM calls.
    """
    homes = glob.glob(os.path.join(dataset_dir, "*"))
    for home in homes:
        process_results(home, client)
