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


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_occluded_images(image_path, num_rows, num_cols):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    H, W, C = img_np.shape
    patch_h = H // num_rows
    patch_w = W // num_cols
    occluded_images = []

    for i in range(num_rows):
        for j in range(num_cols):
            y = i * patch_h
            x = j * patch_w
            occluded = img_np.copy()
            occluded[y:y + patch_h, x:x + patch_w, :] = 0
            occ_img = Image.fromarray(occluded)
            occluded_images.append(((x, y), occ_img))
    return occluded_images


def generate_reverse_occluded_images(image_path, num_rows, num_cols):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    H, W, C = img_np.shape
    patch_h = H // num_rows
    patch_w = W // num_cols
    reverse_occluded_images = []

    base_img = np.zeros_like(img_np)

    for i in range(num_rows):
        for j in range(num_cols):
            y = i * patch_h
            x = j * patch_w
            reverse_occluded = base_img.copy()
            reverse_occluded[y:y + patch_h, x:x + patch_w, :] = img_np[y:y + patch_h, x:x + patch_w, :]
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


def occlusion_test_roof(image_path, client, num_rows, num_cols):
    # Import sentence-transformers utilities
    from sentence_transformers import SentenceTransformer, util
    import torch
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    baseline_text = rate_roof_condition(image_path, client)
    baseline_embedding = embedder.encode(baseline_text, convert_to_tensor=True)
    print(f"Baseline roof rating: {baseline_text}")

    occlusion_results = []
    occlusions = generate_occluded_images(image_path, num_rows, num_cols)

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
                del response_embedding
            else:
                print(f"Skipping patch ({x},{y}) due to invalid rating.")

    import gc
    del embedder, baseline_embedding
    torch.cuda.empty_cache()
    gc.collect()

    return baseline_text, occlusion_results


def reverse_occlusion_test_roof(image_path, client, num_rows, num_cols):
    # Import sentence-transformers utilities
    from sentence_transformers import SentenceTransformer, util
    import torch
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    baseline_text = rate_roof_condition(image_path, client)
    baseline_embedding = embedder.encode(baseline_text, convert_to_tensor=True)
    print(f"Baseline roof rating (reverse occlusion): {baseline_text}")

    occlusion_results = []
    reverse_occlusions = generate_reverse_occluded_images(image_path, num_rows, num_cols)

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
                del response_embedding
            else:
                print(f"Skipping patch ({x},{y}) due to invalid rating.")

    import gc
    del embedder, baseline_embedding
    torch.cuda.empty_cache()
    gc.collect()

    return baseline_text, occlusion_results


def plot_occlusion_heatmap(image_path, occlusion_results, num_rows, num_cols, output_path=None):
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size

    patch_h = img_height // num_rows
    patch_w = img_width // num_cols

    heatmap = np.zeros((num_rows, num_cols))

    for r in occlusion_results:
        x = r["patch_x"]
        y = r["patch_y"]
        i = y // patch_h
        j = x // patch_w
        heatmap[i, j] = r["diff_from_baseline"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Resize heatmap to match image resolution
    heatmap_resized = np.kron(heatmap, np.ones((patch_h, patch_w)))
    ax.imshow(heatmap_resized, cmap='hot', alpha=0.5, interpolation='nearest')

    # Annotate each cell with its difference value
    for row in range(num_rows):
        for col in range(num_cols):
            ax.text(
                col * patch_w + patch_w // 2,
                row * patch_h + patch_h // 2,
                f"{heatmap[row, col]:.2f}",
                ha='center',
                va='center',
                color='black' if heatmap[row, col] < 0.5 else 'white',
                fontsize=8
            )

    ax.set_title("Semantic Sensitivity Heatmap")
    ax.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax, label='Semantic Difference')

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

        good_image = image_dir / f"good_roof_{i}.jpg"
        bad_image = image_dir / f"bad_roof_{i}.jpg"

        baseline_good, results_good = occlusion_test_roof(good_image, client, num_rows=10, num_cols=10)
        plot_occlusion_heatmap(
            good_image,
            results_good,
            10, 10,
            output_path=output_dir / f"good_roof_{i}_heatmap_gpt.png"
        )

        reverse_baseline_good, reverse_results_good = reverse_occlusion_test_roof(good_image, client, num_rows=10, num_cols=10)
        plot_occlusion_heatmap(
            good_image,
            reverse_results_good,
            10, 10,
            output_path=output_dir / f"good_roof_{i}_reverse_heatmap_gpt.png"
        )

        baseline_bad, results_bad = occlusion_test_roof(bad_image, client, num_rows=10, num_cols=10)
        plot_occlusion_heatmap(
            bad_image,
            results_bad,
            10, 10,
            output_path=output_dir / f"bad_roof_{i}_heatmap_gpt.png"
        )

        reverse_baseline_bad, reverse_results_bad = reverse_occlusion_test_roof(bad_image, client, num_rows=10, num_cols=10)
        plot_occlusion_heatmap(
            bad_image,
            reverse_results_bad,
            10, 10,
            output_path=output_dir / f"bad_roof_{i}_reverse_heatmap_gpt.png"
        )
