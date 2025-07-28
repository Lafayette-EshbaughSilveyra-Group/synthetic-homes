from config import BASE_RESULTS_DIR
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import os
import base64
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI


PATCH_SIZE = (50, 50)
STRIDE = (50, 50)


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
            occluded[y:y + patch_size[1], x:x + patch_size[0], :] = 0
            occ_img = Image.fromarray(occluded)
            occluded_images.append(((x, y), occ_img))
    return occluded_images


def generate_reverse_occluded_images(image_path, patch_size, stride):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    H, W, C = img_np.shape
    reverse_occluded_images = []

    base_img = np.zeros_like(img_np)

    for y in range(0, H - patch_size[1] + 1, stride[1]):
        for x in range(0, W - patch_size[0] + 1, stride[0]):
            reverse_occluded = base_img.copy()
            reverse_occluded[y:y + patch_size[1], x:x + patch_size[0], :] = img_np[y:y + patch_size[1], x:x + patch_size[0], :]
            rev_occ_img = Image.fromarray(reverse_occluded)
            reverse_occluded_images.append(((x, y), rev_occ_img))
    return reverse_occluded_images


def rate_roof_condition(image_path, client):
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        print("Skipping invalid or empty image:", image_path)
        return None

    image_base64 = encode_image(image_path)
    prompt = """
You are a certified home inspector. Describe the status roof. Is it in good condition? Why or why not?.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
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
    return reply


def occlusion_test_roof(image_path, client, patch_size=PATCH_SIZE, stride=STRIDE):
    # Import sentence-transformers utilities
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    baseline_text = rate_roof_condition(image_path, client)
    baseline_embedding = embedder.encode(baseline_text, convert_to_tensor=True)
    print(f"Baseline roof rating: {baseline_text}")

    occlusion_results = []
    occlusions = generate_occluded_images(image_path, patch_size, stride)

    for (x, y), occ_img in occlusions:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            occ_img.save(tmp.name)
            response_text = rate_roof_condition(tmp.name, client)
            os.unlink(tmp.name)

            if response_text is not None and baseline_text is not None:
                response_embedding = embedder.encode(response_text, convert_to_tensor=True)
                cosine_sim = util.cos_sim(baseline_embedding, response_embedding).item()
                diff = 1 - cosine_sim
                occlusion_results.append({
                    "patch_x": x,
                    "patch_y": y,
                    "rating": response_text,
                    "diff_from_baseline": diff
                })
                print(f"Patch ({x},{y}) | Diff: {diff:.3f}")
            else:
                print(f"Skipping patch ({x},{y}) due to invalid rating.")

    return baseline_text, occlusion_results


def reverse_occlusion_test_roof(image_path, client, patch_size=PATCH_SIZE, stride=STRIDE):
    # Import sentence-transformers utilities
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    baseline_text = rate_roof_condition(image_path, client)
    baseline_embedding = embedder.encode(baseline_text, convert_to_tensor=True)
    print(f"Baseline roof rating (reverse occlusion): {baseline_text}")

    occlusion_results = []
    reverse_occlusions = generate_reverse_occluded_images(image_path, patch_size, stride)

    for (x, y), rev_occ_img in reverse_occlusions:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            rev_occ_img.save(tmp.name)
            response_text = rate_roof_condition(tmp.name, client)
            os.unlink(tmp.name)

            if response_text is not None and baseline_text is not None:
                response_embedding = embedder.encode(response_text, convert_to_tensor=True)
                cosine_sim = util.cos_sim(baseline_embedding, response_embedding).item()
                diff = 1 - cosine_sim
                occlusion_results.append({
                    "patch_x": x,
                    "patch_y": y,
                    "rating": response_text,
                    "diff_from_baseline": diff
                })
                print(f"Patch ({x},{y}) | Diff: {diff:.3f}")
            else:
                print(f"Skipping patch ({x},{y}) due to invalid rating.")

    return baseline_text, occlusion_results


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
            color=(1, 0, 0, norm_diff),
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


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)

    output_dir = Path(__file__).resolve().parents[3] / BASE_RESULTS_DIR / "experimental" / "occlusion" / "gpt-4.1"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(__file__).resolve().parents[3] / "assets" / "roof_examples"

    for i in range(1, 11):

        print(f"IMAGE {i}")

        good_image = image_dir / f"good_roof_{i}.jpg"
        bad_image = image_dir / f"bad_roof_{i}.jpg"

        print("GOOD")

        baseline_good, results_good = occlusion_test_roof(good_image, client)
        plot_occlusion_heatmap(
            good_image,
            results_good,
            PATCH_SIZE,
            output_path=output_dir / f"good_roof_{i}_heatmap_gpt.png"
        )

        print("GOOD REVERSE")

        reverse_baseline_good, reverse_results_good = reverse_occlusion_test_roof(good_image, client)
        plot_occlusion_heatmap(
            good_image,
            reverse_results_good,
            PATCH_SIZE,
            output_path=output_dir / f"good_roof_{i}_reverse_heatmap_gpt.png"
        )

        print("BAD")

        baseline_bad, results_bad = occlusion_test_roof(bad_image, client)
        plot_occlusion_heatmap(
            bad_image,
            results_bad,
            PATCH_SIZE,
            output_path=output_dir / f"bad_roof_{i}_heatmap_gpt.png"
        )

        print("BAD REVERSE")

        reverse_baseline_bad, reverse_results_bad = reverse_occlusion_test_roof(good_image, client)
        plot_occlusion_heatmap(
            bad_image,
            reverse_results_bad,
            PATCH_SIZE,
            output_path=output_dir / f"bad_roof_{i}_reverse_heatmap_gpt.png"
        )
