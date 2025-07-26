import os.path

import numpy as np
import json
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from shapely.geometry import shape
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from main import generate_geojson_and_note

# ---------- CONFIG ----------

PATCH_SIZE = (50, 50)
STRIDE = (50, 50)


# ---------- UTILS ----------

def encode_image(image_path):
    # Your existing encode function
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_occluded_images(image_path, patch_size, stride):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    H, W, C = img_np.shape
    occluded_images = []

    for y in range(0, H - patch_size[1] + 1, stride[1]):
        for x in range(0, W - patch_size[0] + 1, stride[0]):
            occluded = img_np.copy()
            occluded[y:y + patch_size[1], x:x + patch_size[0], :] = 0  # Black square
            occ_img = Image.fromarray(occluded)
            occluded_images.append(((x, y), occ_img))
    return occluded_images


def compare_geojson(geo1, geo2):
    try:
        poly1 = shape(geo1["features"][0]["geometry"])
        poly2 = shape(geo2["features"][0]["geometry"])
        area_diff = abs(poly1.area - poly2.area) / max(poly1.area, poly2.area, 1e-6)
        iou = poly1.intersection(poly2).area / poly1.union(poly2).area
        iou_diff = 1 - iou
    except Exception as e:
        print("GeoJSON comparison error:", e)
        area_diff, iou_diff = 1.0, 1.0

    props1 = geo1["features"][0]["properties"]
    props2 = geo2["features"][0]["properties"]
    prop_diffs = []
    for key in props1:
        if key in props2:
            if isinstance(props1[key], (int, float)) and isinstance(props2[key], (int, float)):
                val_range = max(abs(props1[key]), abs(props2[key]), 1e-6)
                prop_diffs.append(abs(props1[key] - props2[key]) / val_range)
            else:
                prop_diffs.append(0 if props1[key] == props2[key] else 1)
        else:
            prop_diffs.append(1)
    avg_prop_diff = np.mean(prop_diffs) if prop_diffs else 0

    # Combine metrics to single 0-1 score
    geojson_diff_score = np.clip((area_diff + iou_diff + avg_prop_diff) / 3, 0, 1)
    return geojson_diff_score


def compare_text(text1, text2):
    sm = SequenceMatcher(None, text1, text2)
    return 1 - sm.ratio()  # 0=identical, 1=completely different


# ---------- MAIN TEST FUNCTION ----------

def occlusion_test(house_data, image_path, sketch_path, client, occlude_photo=True):
    from tempfile import NamedTemporaryFile

    # Baseline
    baseline = generate_geojson_and_note(house_data, image_path, sketch_path, client)
    baseline_geojson = baseline["geojson"]
    baseline_note = baseline["inspection_note"]

    # Select image to occlude
    target_path = image_path if occlude_photo else sketch_path
    other_path = sketch_path if occlude_photo else image_path

    occluded_runs = []
    occlusions = generate_occluded_images(target_path, PATCH_SIZE, STRIDE)

    for (x, y), occ_img in occlusions:
        with NamedTemporaryFile(suffix=".jpg" if occlude_photo else ".png", delete=False) as tmp:
            occ_img.save(tmp.name)
            if occlude_photo:
                result = generate_geojson_and_note(house_data, tmp.name, sketch_path, client)
            else:
                result = generate_geojson_and_note(house_data, image_path, tmp.name, client)

        geo_diff = compare_geojson(result["geojson"], baseline_geojson)
        text_diff = compare_text(result["inspection_note"], baseline_note)

        # Combine to overall 0-1 difference score (weighted equally)
        combined_diff = (geo_diff + text_diff) / 2

        occluded_runs.append({
            "patch_x": x,
            "patch_y": y,
            "geo_diff": geo_diff,
            "text_diff": text_diff,
            "combined_diff": combined_diff
        })
        print(f"Patch ({x},{y}) | Geo: {geo_diff:.3f} | Text: {text_diff:.3f} | Combined: {combined_diff:.3f}")

    return occluded_runs


def plot_occlusion_heatmap(image_path, occlusion_results, patch_size, output_path=None):
    """
    image_path: path to the original image
    occlusion_results: list of dicts with keys 'patch_x', 'patch_y', 'combined_diff'
    patch_size: (width, height) tuple
    output_path: if provided, saves the heatmap to this file path
    """
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    ax = plt.gca()

    # Normalize heatmap values
    diffs = [r["combined_diff"] for r in occlusion_results]
    max_diff = max(diffs) if diffs else 1
    min_diff = min(diffs) if diffs else 0

    for r in occlusion_results:
        x = r["patch_x"]
        y = r["patch_y"]
        diff = r["combined_diff"]
        norm_diff = (diff - min_diff) / (max_diff - min_diff + 1e-8)

        rect = plt.Rectangle(
            (x, y),
            patch_size[0],
            patch_size[1],
            color=(1, 0, 0, norm_diff),  # Red heatmap with transparency
            linewidth=1
        )
        ax.add_patch(rect)

    plt.title("Occlusion Sensitivity Heatmap")
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Heatmap saved to {output_path}")
        plt.close()
    else:
        plt.show()


# ---------- USAGE EXAMPLE ----------

HOMES = ["RAMBEAU_RD_15"]

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)  # Initialize your GPT client

    for home in HOMES:
        path = os.path.join('../dataset', home)

        house_data = json.load(open(os.path.join(path, 'data.json')))
        image_path = os.path.join(path, 'photo_1.jpg')
        sketch_path = os.path.join(path, 'sketch.png')

        img_results = occlusion_test(
            house_data,
            image_path,
            sketch_path,
            client=client,
            occlude_photo=True
        )

        sketch_results = occlusion_test(
            house_data,
            image_path,
            sketch_path,
            client=client,
            occlude_photo=False
        )

        # 	•	High opacity / dark red:
        # Occluding this patch significantly changes the GeoJSON or inspection note output.
        # → This region is important for GPT’s generation.
        # 	•	Low opacity / faint red / transparent:
        # Occluding this patch has little to no effect on the outputs.
        # → This region is less influential for the task.

        # For photo occlusion results
        plot_occlusion_heatmap(
            image_path,
            occlusion_results=img_results,
            patch_size=PATCH_SIZE,
            output_path=os.path.join('heatmaps', f"{home}_image_heatmap.png")
        )

        # For sketch occlusion results
        plot_occlusion_heatmap(
            image_path=sketch_path,
            occlusion_results=sketch_results,
            patch_size=PATCH_SIZE,
            output_path=os.path.join('heatmaps', f"{home}_sketch_heatmap.png")
        )
