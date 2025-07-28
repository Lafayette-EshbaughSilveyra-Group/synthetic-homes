"""
geometry_generation.py

Generates building footprint GeoJSON and synthetic inspection notes from property data and images.

Responsibilities:
- Encodes images and floorplan sketches.
- Uses a multimodal model (e.g., GPT-4 Vision, LLaVA) to generate:
    - Building footprint GeoJSON
    - Inspection notes focused on energy efficiency

Pipeline Context:
Step 2. Consumes output from scraper.py.
Produces input for geojson_cleaning.py.

Outputs (per property):
    dataset/{address_folder}/preprocessed.json
"""

import os
import glob
import json
import uuid
from datetime import datetime, timedelta
import base64
import shutil
from typing import Dict, Any
from pipeline.llava_processing import describe_exterior, describe_floorplan


def run_generation_for_dataset(dataset_dir: str, client: Any) -> None:
    """
    Runs geometry generation for all homes in the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory.
        client (Any): API client for LLM calls (OpenAI, LLaVA, etc.).
    """
    for home_folder in glob.glob(os.path.join(dataset_dir, "*")):
        try:
            print(f"[→] Generating for {home_folder}")
            result = generate_geojson_and_note(
                json.load(open(os.path.join(home_folder, "data.json"))),
                os.path.join(home_folder, "photo_1.jpg"),
                os.path.join(home_folder, "sketch.png"),
                client
            )
            json.dump(result, open(os.path.join(home_folder, "preprocessed.json"), "w", encoding='utf-8'), indent=2)
            print(f"[GENERATED] {home_folder}")
        except Exception as e:
            print(f"[FAILED] {home_folder} [WILL BE DELETED]: {e}")
            shutil.rmtree(home_folder, ignore_errors=True)


def encode_image(filepath: str) -> str:
    """
    Encodes an image file to base64.

    Args:
        filepath (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_geojson_and_note(house_data: Dict[str, Any], image_path: str, sketch_path: str, client: Any) -> Dict[str, Any]:
    """
    Calls a multimodal model to generate GeoJSON and an inspection note.

    Args:
        house_data (Dict): Structured property data.
        image_path (str): Path to exterior image.
        sketch_path (str): Path to sketch image.
        client (Any): API client for LLM calls.

    Returns:
        Dict: {
            "geojson": { ... },
            "inspection_note": "..."
        }
    """
    exterior_description = describe_exterior(image_path)
    floorplan_description = describe_floorplan(sketch_path)

    # ----- Prompt Setup -----
    prompt = f"""
    You are a certified home energy inspection expert and data specialist building synthetic training data for an AI model.

    You are provided with:
    - Structured residential property data (JSON).
    - A detailed exterior description of the home: "{exterior_description}"
    - A detailed floorplan description: "{floorplan_description}"

    Your tasks:
    1️⃣ Generate a **GeoJSON file** for this building with:
    - A plausible (longitude, latitude) location in Bethlehem, PA.
    - A "FeatureCollection" containing exactly **one Feature**.
    - Geometry: Polygon or MultiPolygon representing the building footprint, realistic for the reported square footage and building style.
    - Properties from the provided JSON:
        - "Year Built"
        - "Total Square Feet Living Area"
        - "Building Style"
        - "Exterior Wall Material"
        - "Heating Fuel Type"
        - "Heating System Type"
        - "Heat/Air Cond"
        - "Bedrooms"
        - "Full Baths"
        - "Half Baths"
        - "Basement"
        - "Number of Stories"
        - "Grade"
    - Estimated performance parameters:
        - "hvac_heating_cop" (0-1)
        - "hvac_cooling_cop"
        - "wall_r_value"
        - "roof_r_value"
        - "air_change_rate" (0-1)

    2️⃣ Write a short **inspection note** as if you had toured the home, focusing on energy-related observations: insulation, HVAC type/age, visible windows, and any inferred upgrades.

    **Strict Guidelines:**
    - Only base your outputs on the provided data and descriptions.
    - Do not invent details not clearly supported by the inputs.
    - Ensure the GeoJSON is valid and realistic.
    - Coordinates should place the home plausibly in Bethlehem, PA.

    Here is the structured property data:

    {json.dumps(house_data)}

    **Output Format:**
    Return a raw JSON object:
    {{
      "geojson": {{
        "type": "FeatureCollection",
        "features": [
          {{
            "type": "Feature",
            "geometry": {{ ... }},
            "properties": {{
              ...,
              "air_change_rate": ...,
              "hvac_heating_cop": ...,
              "hvac_cooling_cop": ...,
              "wall_r_value": ...,
              "roof_r_value": ...,
            }}
          }}
        ]
      }},
      "inspection_note": "..."
    }}
    No backticks or explanation.
    """

    # ----- API Call -----
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    # ----- Parse -----
    reply = response.choices[0].message.content
    return json.loads(reply)


def clean_gpt_geojson(gpt_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and formats the GPT-generated GeoJSON into final schema.

    Args:
        gpt_output (Dict): Raw output from LLM inference.

    Returns:
        Dict: Cleaned GeoJSON object.
    """
    def safe_int(val):
        try:
            return int(str(val).replace(",", "").strip())
        except:
            return None

    end_date = datetime.now()
    begin_date = end_date - timedelta(days=365)

    begin_date_str = begin_date.strftime('%Y-%m-%dT00:00:00Z')
    end_date_str = end_date.strftime('%Y-%m-%dT00:00:00Z')

    full_geojson = {
        "type": "FeatureCollection",
        "mappers": [],
        "project": {
            "id": str(uuid.uuid4()),
            "name": "Generated Project",
            "begin_date": begin_date_str,
            "end_date": end_date_str,
            "cec_climate_zone": None,
            "climate_zone": "4A",
            "default_template": "90.1-2013",
            "import_surrounding_buildings_as_shading": None,
            "surface_elevation": None,
            "tariff_filename": None,
            "timesteps_per_hour": 1,
            "weather_filename": "weather.epw"
        },
        "scenarios": [
            {
                "feature_mappings": [],
                "id": str(uuid.uuid4()),
                "name": "Base Scenario"
            }
        ],
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": str(uuid.uuid4()),
                    "name": "Generated Home",
                    "type": "Building",
                    "building_type": "Single family",
                    "floor_area": safe_int(gpt_output["geojson"]["properties"].get("Total Square Feet Living Area")),
                    "number_of_stories": safe_int(gpt_output["geojson"]["properties"].get("Number of Stories")),
                    "inspection_note": gpt_output["inspection_note"],
                    **gpt_output["geojson"]["properties"]
                },
                "geometry": gpt_output["geojson"]["geometry"]
            }
        ]
    }

    return full_geojson


def clean_gpt_geojson_for_all_entries(dataset_dir: str = 'dataset') -> None:
    """
    Cleans and formats GeoJSON for all entries in the dataset.

    Args:
        dataset_dir (str): Path to the dataset directory.
    """
    home_folders = glob.glob(os.path.join(dataset_dir, '*'))
    for home_folder in home_folders:
        for attempt in range(5):
            try:
                result = clean_gpt_geojson(json.load(open(os.path.join(home_folder, "preprocessed.json"))))
                json.dump(result, open(os.path.join(home_folder, "cleaned.geojson"), "w", encoding='utf-8'), indent=2)
                break
            except Exception as e:
                if attempt == 4:
                    print(f"[CLEANING FAILED] {home_folder} [WILL BE DELETED]: {e}")
                    shutil.rmtree(home_folder, ignore_errors=True)
                else:
                    print(f"[RETRY {attempt + 1}/5] {home_folder} failed: {e}")
