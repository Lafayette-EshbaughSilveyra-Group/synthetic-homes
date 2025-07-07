import base64
import io
import json
import traceback

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from shapely.geometry import shape
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# ================================
# 🔧 1. CONFIGURATION
# ================================

GRID_SIZE = 4  # Grid granularity (10x10 tiles)
MASK_COLOR = (0, 0, 0)  # Black mask color
RUN_HOMES = ["RAMBEAU_RD_15"]  # Replace with your home IDs
DATASET_DIR = "../dataset"  # Base directory of images
OUTPUT_DIR = "heatmaps"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# 🔧 2. HELPER FUNCTIONS
# ================================

def mask_tile(image, tile_coords, tile_size):
    masked_image = image.copy()
    x, y = tile_coords
    mask_box = [x, y, x + tile_size[0], y + tile_size[1]]
    masked_image.paste(MASK_COLOR, mask_box)
    return masked_image


# Load your embedding model once globally
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def compute_difference_metric(output1, output2):
    """
    Computes a numeric difference metric between two outputs.

    - If outputs are geojson with 'geometry' key, computes 1 - IoU.
    - If outputs are text with 'inspection_note' key, computes embedding cosine distance.

    Returns a float: higher = more different.
    """

    # Check for geojson polygon outputs
    if 'geojson' in output1 and 'geojson' in output2:
        try:
            feature1 = output1['geojson']['features'][0]
            feature2 = output2['geojson']['features'][0]

            poly1 = shape(feature1['geometry'])
            poly2 = shape(feature2['geometry'])

            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area

            if union == 0:
                return 1.0

            iou = intersection / union
            return 1 - iou

        except Exception as e:
            print(f"GeoJSON difference metric error: {e}")
            return 1.0

    # Check for textual outputs
    elif 'inspection_note' in output1 and 'inspection_note' in output2:
        try:
            text1 = output1['inspection_note']
            text2 = output2['inspection_note']

            emb1 = embedder.encode(text1)
            emb2 = embedder.encode(text2)

            print("Text1:", text1)
            print("Text2:", text2)
            print("Emb1:", emb1)
            print("Emb2:", emb2)
            print("Cosine distance:", cosine(emb1, emb2))

            return cosine(emb1, emb2)

        except Exception as e:
            print(f"Text difference metric error: {e}")
            traceback.print_exc()
            return 1.0

    else:
        print("Unknown output structure for difference metric.")
        return 1.0


def run_ablation_heatmap_image(house_data, image, sketch, grid_size=GRID_SIZE):
    width, height = image.size
    tile_w, tile_h = width // grid_size, height // grid_size

    baseline_output = model_generate_fn(house_data, image, sketch)

    heatmap = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            x, y = i * tile_w, j * tile_h
            tile_size = (tile_w, tile_h)
            masked_img = mask_tile(image, (x, y), tile_size)

            output = model_generate_fn(house_data, masked_img, sketch)
            diff = compute_difference_metric(baseline_output, output)
            heatmap[j, i] = diff

            print(f"Tile ({i}, {j}) – Difference: {diff:.4f}")

    return heatmap


def run_ablation_heatmap_sketch(house_data, sketch, image, grid_size=GRID_SIZE):
    width, height = sketch.size
    tile_w, tile_h = width // grid_size, height // grid_size

    baseline_output = model_generate_fn(house_data, image, sketch)

    heatmap = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            x, y = i * tile_w, j * tile_h
            tile_size = (tile_w, tile_h)
            masked_img = mask_tile(sketch, (x, y), tile_size)

            output = model_generate_fn(house_data, image, masked_img)
            diff = compute_difference_metric(baseline_output, output)
            heatmap[j, i] = diff

            print(f"Tile ({i}, {j}) – Difference: {diff:.4f}")

    return heatmap


def plot_and_save_heatmap(heatmap, original_image, save_base_path, title="Saliency Heatmap"):
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='hot', alpha=0.5, extent=(0, original_image.size[0], original_image.size[1], 0))
    plt.title(title)
    plt.colorbar(label='Importance')
    plt.axis('off')

    plt.savefig(f"{save_base_path}_overlay.png", bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap overlay to {save_base_path}_overlay.png")

    # Save heatmap array as .npy
    np.save(f"{save_base_path}.npy", heatmap)
    print(f"Saved heatmap array to {save_base_path}.npy")


# ================================
# 🔧 3. YOUR GENERATION FUNCTION
# ================================

def model_generate_fn(house_data, image, sketch):
    def encode_image(img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def safe_json_parse(raw_content):
        """
        Cleans and parses raw model output as JSON.
        """
        # Remove code block formatting if present
        json_str = re.sub(r"^```json\s*|```$", "", raw_content.strip(), flags=re.MULTILINE)

        try:
            return json.loads(json_str)
        except Exception as e:
            print("⚠️ Failed to parse JSON. Raw content below:")
            print(raw_content)
            raise e

    image_base64 = encode_image(image)
    sketch_base64 = encode_image(sketch)

    # =============== 🔥 Stage 1: GeoJSON + base inspection note generation ===============
    stage1_prompt = f"""
        You are a certified **home energy inspection expert**.

        You are given:
        - Structured residential property data in JSON format
        - A sketch of the floorplan of the home

        Structured property data:

        {json.dumps(house_data)}

        Use these to generate:

        1. A **GeoJSON File** describing the home footprint as a valid FeatureCollection with exactly one Feature, with:
           - Plausible location in Bethlehem, PA
           - Geometry representing the structural footprint (polygon or multipolygon)
           - Properties including:
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
               - "air_change_rate" (ACH)
               - "hvac_heating_cop"
               - "hvac_cooling_cop"
               - "wall_r_value"
               - "roof_r_value"

        2. A **base inspection note** summarizing general energy characteristics inferred from the structured data and sketch.

        Return as:

        {{
          "geojson": {{ ... }},
          "inspection_note": "..."
        }}

        Output **ONLY valid raw JSON without any explanation, commentary, or markdown code block formatting**. Your response must start with '{{' and end with '}}'.
    """

    stage1_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": stage1_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sketch_base64}"}},
                ]
            }
        ],
        temperature=0.7
    )

    stage1_output = safe_json_parse(stage1_response.choices[0].message.content)

    # =============== 🔥 Stage 2: Enrichment with exterior photo details ===============
    base_inspection_note = stage1_output["inspection_note"]

    stage2_prompt = f"""
        You are a certified **home energy inspection expert**.

        You previously wrote the following inspection note:

        "{base_inspection_note}"

        Now, enrich and expand this note based on the **exterior photo** provided. Focus on visible energy-relevant details including:

        - Window frame material and condition
        - Number of visible windows
        - Roof type and condition
        - Siding material (if visible)
        - Exterior HVAC units or vents
        - Presence of shading (trees, awnings)
        - Any observable damage, weathering, or upgrades

        Return ONLY the improved inspection note as:

        {{
          "enriched_inspection_note": "..."
        }}

        Output **ONLY valid raw JSON without any explanation, commentary, or markdown code block formatting**. Your response must start with '{{' and end with '}}'.
    """

    stage2_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": stage2_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ]
            }
        ],
        temperature=0.7
    )

    stage2_output = safe_json_parse(stage2_response.choices[0].message.content)

    # =============== 🔥 Combine outputs ===============
    stage1_output["inspection_note"] = stage2_output["enriched_inspection_note"]

    return stage1_output


# ================================
# 🔧 4. MAIN BATCH PROCESSING
# ================================

if __name__ == "__main__":
    for home_id in RUN_HOMES:
        print(f"\n=== Processing {home_id} ===")

        # 📷 Load image
        image_path = os.path.join(DATASET_DIR, home_id, "photo_1.jpg")
        image = Image.open(image_path).convert("RGB")

        sketch_path = os.path.join(DATASET_DIR, home_id, "sketch.png")
        sketch = Image.open(sketch_path).convert("RGB")

        house_data = json.load(open(os.path.join(DATASET_DIR, home_id, "data.json"), "r"))

        # 🔥 Run ablation heatmap
        heatmap = run_ablation_heatmap_image(house_data, image, sketch, grid_size=GRID_SIZE)

        # 🎨 Plot and save
        save_base_path = os.path.join(OUTPUT_DIR, f"{home_id}_image_heatmap")
        plot_and_save_heatmap(heatmap, image, save_base_path, title=f"{home_id} Saliency Heatmap")

        # 🔥 Run ablation heatmap
        heatmap = run_ablation_heatmap_sketch(house_data, sketch, image, grid_size=GRID_SIZE)

        # 🎨 Plot and save
        save_base_path = os.path.join(OUTPUT_DIR, f"{home_id}_sketch_heatmap")
        plot_and_save_heatmap(heatmap, sketch, save_base_path, title=f"{home_id} Sketch Saliency Heatmap")
