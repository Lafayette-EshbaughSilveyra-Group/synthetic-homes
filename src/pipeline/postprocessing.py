import os
import glob
import json
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


def label_data(results_json: dict, inspection_report: str, home_dir_name: str, client: Any,
               text_weight: float = 0.2, energyplus_weight: float = 0.8, energyplus_label_method: str = 'heuristic') -> \
        dict[str, float | Any]:
    """
    Uses OpenAI API to label a datapoint based on its results.json and inspection report.
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
        heating_load_hourly_J = 0.0
        hvac_hourly_J = 0.0

        for var_name, var_data in results.items():
            if "Heating Coil Heating Energy" in var_name:
                heating_load_hourly_J = float(var_data["mean"])
            elif "Electricity:HVAC" in var_name:
                hvac_hourly_J = float(var_data["mean"])

        heating_load_annual_kWh = (heating_load_hourly_J * 730 * 12) / 3600000
        hvac_annual_kWh = (hvac_hourly_J * 730 * 12) / 3600000

        insulation_score = (heating_load_annual_kWh - config.HL_BEST) / (config.HL_WORST - config.HL_BEST)
        hvac_score = (hvac_annual_kWh - config.HVAC_BEST) / (config.HVAC_WORST - config.HVAC_BEST)

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
        text_data = json.loads(text_content)

        if energyplus_label_method == 'gpt':
            energyplus_response = safe_chat_response(energyplus_prompt)
            energyplus_content = extract_json_from_response(energyplus_response.choices[0].message.content)
            energyplus_data = json.loads(energyplus_content)
        elif energyplus_label_method == 'heuristic':
            energyplus_data = heuristic_labeler(results_json)
        else:
            raise ValueError(f"Unsupported energyplus label method: {energyplus_label_method}")

        result = {
            "insulation": ((text_data["insulation"] * text_weight) + (
                        energyplus_data["insulation"] * energyplus_weight)) / 2,
            "hvac": ((text_data["hvac"] * text_weight) + (energyplus_data["hvac"] * energyplus_weight)) / 2,
        }

        if home_dir_name is None:
            return result
        label_path = f"{home_dir_name}/label.json"
        with open(label_path, "w") as f:
            json.dump(result, f, indent=2)

    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON. Raw content:")
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
