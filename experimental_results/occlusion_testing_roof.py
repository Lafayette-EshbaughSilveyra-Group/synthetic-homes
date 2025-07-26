import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import os
import base64

from dotenv import load_dotenv

# ---------- CONFIG ----------

PATCH_SIZE = (50, 50)
STRIDE = (50, 50)


# ---------- UTILS ----------

def encode_image(image_path):
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


# ---------- GPT ROOF RATING FUNCTION ----------

def rate_roof_condition(image_path, client):
    """
    Prompts GPT to rate roof condition from 0 (poor) to 1 (excellent).
    Returns float rating or None on failure.
    """
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        print("Skipping invalid or empty image:", image_path)
        return None

    image_base64 = encode_image(image_path)

    prompt = """
You are a certified home inspector. Rate the condition of the roof in the provided photo from 0 (extremely poor condition) to 1 (excellent condition). Output only a single number between 0 and 1.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Replace with your vision model if needed
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ]
                }
            ],
            temperature=0
        )
    except Exception as e:
        print("GPT request failed:", e)
        return None

    reply = response.choices[0].message.content.strip()
    try:
        rating = float(reply)
        return max(0.0, min(1.0, rating))  # Clamp to [0,1]
    except ValueError:
        print("Invalid rating returned:", reply)
        return None


# ---------- OCCLUSION TEST FUNCTION ----------

def occlusion_test_roof(image_path, client, patch_size=PATCH_SIZE, stride=STRIDE):
    baseline_rating = rate_roof_condition(image_path, client)
    print(f"Baseline roof rating: {baseline_rating}")

    occlusion_results = []
    occlusions = generate_occluded_images(image_path, patch_size, stride)

    for (x, y), occ_img in occlusions:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            occ_img.save(tmp.name)
            rating = rate_roof_condition(tmp.name, client)
            os.unlink(tmp.name)  # Delete temp file immediately

            if rating is not None and baseline_rating is not None:
                diff = abs(rating - baseline_rating)
                occlusion_results.append({
                    "patch_x": x,
                    "patch_y": y,
                    "rating": rating,
                    "diff_from_baseline": diff
                })
                print(f"Patch ({x},{y}) | Rating: {rating:.3f} | Diff: {diff:.3f}")
            else:
                print(f"Skipping patch ({x},{y}) due to invalid rating.")

    return baseline_rating, occlusion_results


# ---------- HEATMAP VISUALIZATION ----------

def plot_occlusion_heatmap(image_path, occlusion_results, patch_size, output_path=None):
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    ax = plt.gca()

    diffs = [r["diff_from_baseline"] for r in occlusion_results]
    max_diff = max(diffs) if diffs else 1e-6
    min_diff = min(diffs) if diffs else 0

    for r in occlusion_results:
        x = r["patch_x"]
        y = r["patch_y"]
        diff = r["diff_from_baseline"]
        norm_diff = (diff - min_diff) / (max_diff - min_diff + 1e-8)

        rect = plt.Rectangle(
            (x, y),
            patch_size[0],
            patch_size[1],
            color=(1, 0, 0, norm_diff),  # Red heatmap with transparency
            linewidth=1
        )
        ax.add_patch(rect)

    plt.title("Roof Rating Occlusion Sensitivity Heatmap")
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Heatmap saved to {output_path}")
        plt.close()
    else:
        plt.show()


# ---------- USAGE EXAMPLE ----------

if __name__ == "__main__":
    from openai import OpenAI

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)

    os.makedirs('roof_occ_tests', exist_ok=True)

    # Example good roof image
    good_image = "good_roof.jpg"
    baseline_good, results_good = occlusion_test_roof(good_image, client)
    plot_occlusion_heatmap(good_image, results_good, PATCH_SIZE, output_path="roof_occ_tests/good_roof_heatmap.png")

    # Example broken roof image
    broken_image = "bad_roof.jpg"
    baseline_broken, results_broken = occlusion_test_roof(broken_image, client)
    plot_occlusion_heatmap(broken_image, results_broken, PATCH_SIZE,
                           output_path="roof_occ_tests/bad_roof_heatmap.png")