import json
import tiktoken

# Load your experimental set
dataset = json.load(open('energyplus_data/experimental_set.json', 'r'))

# Load tiktoken encoder for your model
enc = tiktoken.encoding_for_model("gpt-4o")

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
  \"hvac\": 0.5
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

{ json.dumps(results_json, indent=2) }

### RESPONSE FORMAT:
Return a JSON object like:
{{
  \"insulation\": 0.5,
  \"hvac\": 0.5
}}

Only include the JSON. No explanation or commentary.
"""

# Example inspection text sample (use your full list as needed)
inspection_text = "There is an older HVAC unit installed, with signs of rust on the exterior."

# Analyze for each example in dataset
for key, example in dataset.items():
    text_prompt = build_text_prompt(inspection_text)
    eplus_prompt = build_energyplus_prompt(example)

    text_tokens = len(enc.encode(text_prompt))
    eplus_tokens = len(enc.encode(eplus_prompt))

    print(f"Example: {key}")
    print(f"  Text prompt tokens: {text_tokens}")
    print(f"  EnergyPlus prompt tokens: {eplus_tokens}")
    print(f"  Total tokens (no output included): {text_tokens + eplus_tokens}")
    print("-" * 50)