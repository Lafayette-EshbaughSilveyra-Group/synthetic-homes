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
from copy import deepcopy
from typing import Dict, Any
from pipeline.llava_processing import describe_exterior, describe_floorplan
from pyproj import Geod


def safe_float(val, default=None):
    """
    Safely converts values to float.
    """
    try:
        if val is None:
            return default
        return float(str(val).replace(",", "").strip())
    except:
        return default


def estimate_target_footprint_area(house_data):
    """
    Estimates target building footprint area in ft².

    Uses:
        footprint_area ≈ total_floor_area / number_of_stories
    """

    floor_area = safe_float(
        house_data.get("Total Square Feet Living Area")
    )

    stories = safe_float(
        house_data.get("Number of Stories"),
        default=1
    )

    if floor_area is None or floor_area <= 0:
        return None

    if stories is None or stories <= 0:
        stories = 1

    return floor_area / stories


def compute_polygon_area_ft2(geometry):
    """
    Computes GeoJSON polygon area in ft².

    Supports:
    - Polygon
    - MultiPolygon
    """

    geod = Geod(ellps='WGS84')

    def polygon_area(coords):
        ring = coords[0]

        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]

        area_m2, _ = geod.polygon_area_perimeter(lons, lats)

        area_m2 = abs(area_m2)

        return area_m2 * 10.76391  # m² → ft²

    geom_type = geometry.get("type")

    if geom_type == "Polygon":
        return polygon_area(geometry["coordinates"])

    elif geom_type == "MultiPolygon":
        total = 0

        for polygon in geometry["coordinates"]:
            total += polygon_area(polygon)

        return total

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def scale_geometry_to_target_area(geometry, house_data):
    """
    Deterministically rescales a generated GeoJSON footprint around its centroid
    so that its area is closer to the metadata-derived target footprint area.

    This preserves the model-generated shape approximately while correcting
    unrealistic coordinate scale.
    """

    target_area = estimate_target_footprint_area(house_data)

    if target_area is None or target_area <= 0:
        return geometry

    current_area = compute_polygon_area_ft2(geometry)

    if current_area <= 0:
        return geometry

    scale_factor = (target_area / current_area) ** 0.5
    scaled_geometry = deepcopy(geometry)

    def collect_points_from_polygon(coords):
        points = []
        for ring in coords:
            points.extend(ring)
        return points

    geom_type = scaled_geometry.get("type")

    if geom_type == "Polygon":
        all_points = collect_points_from_polygon(scaled_geometry["coordinates"])
    elif geom_type == "MultiPolygon":
        all_points = []
        for polygon in scaled_geometry["coordinates"]:
            all_points.extend(collect_points_from_polygon(polygon))
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    if not all_points:
        return geometry

    center_lon = sum(point[0] for point in all_points) / len(all_points)
    center_lat = sum(point[1] for point in all_points) / len(all_points)

    def scale_ring(ring):
        scaled_ring = []
        for lon, lat in ring:
            new_lon = center_lon + (lon - center_lon) * scale_factor
            new_lat = center_lat + (lat - center_lat) * scale_factor
            scaled_ring.append([new_lon, new_lat])

        if scaled_ring and scaled_ring[0] != scaled_ring[-1]:
            scaled_ring.append(scaled_ring[0])

        return scaled_ring

    if geom_type == "Polygon":
        scaled_geometry["coordinates"] = [scale_ring(ring) for ring in scaled_geometry["coordinates"]]
    elif geom_type == "MultiPolygon":
        scaled_geometry["coordinates"] = [
            [scale_ring(ring) for ring in polygon]
            for polygon in scaled_geometry["coordinates"]
        ]

    return scaled_geometry


def geometry_is_reasonable(
    geometry,
    house_data,
    lower_ratio=0.4,
    upper_ratio=2.5
):
    """
    Checks whether generated geometry footprint area is
    reasonably consistent with estimated target footprint area.

    Returns:
        (bool, dict)
    """

    target_area = estimate_target_footprint_area(house_data)

    if target_area is None:
        return True, {
            "reason": "missing_target_area"
        }

    generated_area = compute_polygon_area_ft2(geometry)

    ratio = generated_area / target_area

    is_valid = lower_ratio <= ratio <= upper_ratio

    return is_valid, {
        "target_area_ft2": target_area,
        "generated_area_ft2": generated_area,
        "ratio": ratio
    }

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

    target_footprint_area = estimate_target_footprint_area(house_data)
    target_footprint_text = (
        f"{target_footprint_area:.1f} square feet"
        if target_footprint_area is not None
        else "unknown; estimate conservatively from the floorplan description"
    )

    # ----- Prompt Setup -----
    prompt = f"""
    You are a certified home energy inspection expert and data specialist building synthetic training data for an AI model.

    You are provided with:
    - Structured residential property data (JSON).
    - A detailed exterior description of the home: "{exterior_description}"
    - A detailed floorplan description: "{floorplan_description}"

    Your tasks:
    1. Generate a **GeoJSON file** for this building with:
    - A plausible (longitude, latitude) location in Bethlehem, PA.
    - A "FeatureCollection" containing exactly **one Feature**.
    - Geometry: Polygon or MultiPolygon representing an approximate simulation-ready building footprint.
    - The geometry should be realistic in scale for the reported square footage, number of stories, and building style.
    - Use the floorplan description to infer approximate shape, but use the structured property data to control scale.
    - Target approximate footprint area: {target_footprint_text}.
    - The generated polygon should represent the building footprint, not the parcel or lot boundary.
    - The polygon area should be within approximately 20-30% of the target footprint area when possible.
    - Avoid large coordinate spans. For typical residential homes in Bethlehem, PA, longitude/latitude differences should usually be very small.
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

    2. Write a short **inspection note** as if you had toured the home, focusing on energy-related observations: insulation, HVAC type/age, visible windows, and any inferred upgrades.

    **Strict Guidelines:**
    - Only base your outputs on the provided data and descriptions.
    - Do not invent details not clearly supported by the inputs.
    - Ensure the GeoJSON is valid and realistic.
    - Coordinates should place the home plausibly in Bethlehem, PA.
    - Do not generate parcel-sized rectangles.
    - Do not use coordinate differences such as 0.0005 degrees unless the resulting footprint area is consistent with the target footprint area.
    - Prefer compact residential-scale polygons.

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

    def call_generation_model(current_prompt: str) -> Dict[str, Any]:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": current_prompt
                }
            ],
            temperature=0.7
        )

        reply = response.choices[0].message.content
        return json.loads(reply)

    def get_generated_geometry(parsed_output: Dict[str, Any]):
        return (
            parsed_output
            .get("geojson", {})
            .get("features", [{}])[0]
            .get("geometry")
        )

    # ----- API Call + Geometry Validation -----
    max_generation_attempts = 6
    last_error = None

    for attempt in range(max_generation_attempts):
        current_prompt = prompt

        if attempt > 0 and last_error is not None:
            current_prompt = f"""
            {prompt}

            IMPORTANT CORRECTION:
            Your previous GeoJSON geometry was rejected because its footprint area was not realistic in scale.

            Target footprint area: {last_error.get('target_area_ft2', 'unknown')} ft²
            Generated footprint area: {last_error.get('generated_area_ft2', 'unknown')} ft²
            Generated/target area ratio: {last_error.get('ratio', 'unknown')}

            If the generated/target area ratio is greater than 1, the footprint is too large; reduce the coordinate span.
            If the generated/target area ratio is less than 1, the footprint is too small; increase the coordinate span.
            Adjust the footprint scale while preserving a compact residential building shape.

            Regenerate the full raw JSON object, but correct the GeoJSON geometry so that:
            - the footprint area is approximately consistent with the target footprint area,
            - the geometry represents the building footprint, not the parcel or lot boundary,
            - the coordinates remain plausible for Bethlehem, PA,
            - the output remains valid raw JSON with the same top-level structure.
            """

        parsed = call_generation_model(current_prompt)
        geometry = get_generated_geometry(parsed)

        if not geometry:
            last_error = {"reason": "missing_geometry"}
            continue

        valid, info = geometry_is_reasonable(
            geometry,
            house_data
        )

        if valid:
            return parsed

        last_error = info
        print(
            f"[GEOMETRY RETRY {attempt + 1}/{max_generation_attempts}] "
            f"Generated footprint unreasonable: {info}"
        )

    print(
        f"[GEOMETRY WARNING] Geometry remained outside target bounds after "
        f"{max_generation_attempts} attempts; applying deterministic scale correction: {last_error}"
    )

    geometry = get_generated_geometry(parsed)

    if not geometry:
        raise ValueError(
            f"Generated geometry missing after {max_generation_attempts} attempts: {last_error}"
        )

    scaled_geometry = scale_geometry_to_target_area(geometry, house_data)

    parsed["geojson"]["features"][0]["geometry"] = scaled_geometry

    valid, info = geometry_is_reasonable(
        scaled_geometry,
        house_data
    )

    if not valid:
        raise ValueError(
            f"Generated geometry remained unreasonable after corrective scaling: {info}"
        )

    print(f"[GEOMETRY SCALED] Corrected generated footprint scale: {info}")

    return parsed


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

    feature = gpt_output.get("geojson", {}).get("features", [{}])[0]
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})

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
                    "floor_area": safe_int(props.get("Total Square Feet Living Area")),
                    "number_of_stories": safe_int(props.get("Number of Stories")),
                    "inspection_note": gpt_output.get("inspection_note", ""),
                    **props
                },
                "geometry": geom
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
