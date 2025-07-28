import torch
from config import BASE_RESULTS_DIR
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\nYou are a certified home inspector. Describe the status roof. Is it in good condition? Why or why not? ASSISTANT:"
GRID_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCCLUSION_COLOR = (127, 127, 127)


model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")


def run_inference(image, text_prompt):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


def mask_image(img, mask_box, color=OCCLUSION_COLOR):
    masked = img.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle(mask_box, fill=color)
    return masked


def get_occlusion_boxes(img_width, img_height, grid_size):
    patch_w = img_width // grid_size
    patch_h = img_height // grid_size
    boxes = [(i * patch_w, j * patch_h, (i + 1) * patch_w, (j + 1) * patch_h)
             for i in range(grid_size) for j in range(grid_size)]
    return boxes


def run_occlusion_test_with_heatmap(image_path, heatmap_filename):
    original_img = Image.open(image_path).convert("RGB")
    img_width, img_height = original_img.size
    boxes = get_occlusion_boxes(img_width, img_height, GRID_SIZE)

    print(f"Running occlusion test on {len(boxes)} patches ({GRID_SIZE}x{GRID_SIZE} grid).")

    original_output = run_inference(original_img, PROMPT)
    original_embedding = embedder.encode(original_output, convert_to_tensor=True)
    print(f"\nOriginal output:\n{original_output}\n")

    heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

    for idx, box in enumerate(boxes):
        occluded_img = mask_image(original_img, box)
        output_text = run_inference(occluded_img, PROMPT)
        occluded_embedding = embedder.encode(output_text, convert_to_tensor=True)
        cosine_sim = util.cos_sim(original_embedding, occluded_embedding).item()
        diff = 1 - cosine_sim

        i, j = idx % GRID_SIZE, idx // GRID_SIZE
        heatmap[j, i] = diff

        print(f"\n--- Patch {idx + 1}/{len(boxes)} ---")
        print(f"Mask box: {box}")
        print(f"Cosine similarity: {cosine_sim:.4f} (Difference: {diff:.4f})")
        print(f"Output: {output_text}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_img)
    heatmap_resized = np.kron(heatmap, np.ones((img_height // GRID_SIZE, img_width // GRID_SIZE)))
    ax.imshow(heatmap_resized, cmap='hot', alpha=0.5, interpolation='nearest')

    ax.set_title("Occlusion Sensitivity Heatmap (1 - Cosine Similarity)")
    ax.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax, label='Semantic Difference')
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    plt.close()


def run_reverse_occlusion_test_with_heatmap(image_path, heatmap_filename):
    original_img = Image.open(image_path).convert("RGB")
    img_width, img_height = original_img.size
    boxes = get_occlusion_boxes(img_width, img_height, GRID_SIZE)

    print(f"Running REVERSE occlusion test on {len(boxes)} patches ({GRID_SIZE}x{GRID_SIZE} grid).")

    original_output = run_inference(original_img, PROMPT)
    original_embedding = embedder.encode(original_output, convert_to_tensor=True)
    print(f"\nOriginal output:\n{original_output}\n")

    fully_masked = Image.new('RGB', original_img.size, OCCLUSION_COLOR)
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

    for idx, box in enumerate(boxes):
        temp_img = fully_masked.copy()
        temp_img.paste(original_img.crop(box), box)

        output_text = run_inference(temp_img, PROMPT)
        occluded_embedding = embedder.encode(output_text, convert_to_tensor=True)
        cosine_sim = util.cos_sim(original_embedding, occluded_embedding).item()
        diff = 1 - cosine_sim

        i, j = idx % GRID_SIZE, idx // GRID_SIZE
        heatmap[j, i] = diff

        print(f"\n--- Patch {idx + 1}/{len(boxes)} ---")
        print(f"Reveal box: {box}")
        print(f"Cosine similarity: {cosine_sim:.4f} (Difference: {diff:.4f})")
        print(f"Output: {output_text}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_img)
    heatmap_resized = np.kron(heatmap, np.ones((img_height // GRID_SIZE, img_width // GRID_SIZE)))
    ax.imshow(heatmap_resized, cmap='hot', alpha=0.5, interpolation='nearest')

    ax.set_title("Reverse Occlusion Heatmap (1 - Cosine Similarity)")
    ax.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax, label='Semantic Difference')
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    output_dir = Path(__file__).resolve().parents[3] / BASE_RESULTS_DIR / "experimental" / "occlusion" / "llava"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_occlusion_test_with_heatmap(
        Path(__file__).resolve().parents[3] / "assets" / "roof_examples" / "good_roof.jpg",
        output_dir / "good_roof_occlusion_heatmap_llava.png"
    )
    run_occlusion_test_with_heatmap(
        Path(__file__).resolve().parents[3] / "assets" / "roof_examples" / "bad_roof.jpg",
        output_dir / "bad_roof_occlusion_heatmap_llava.png"
    )
    run_reverse_occlusion_test_with_heatmap(
        Path(__file__).resolve().parents[3] / "assets" / "roof_examples" / "good_roof.jpg",
        output_dir / "good_roof_reverse_occlusion_heatmap_llava.png"
    )
    run_reverse_occlusion_test_with_heatmap(
        Path(__file__).resolve().parents[3] / "assets" / "roof_examples" / "bad_roof.jpg",
        output_dir / "bad_roof_reverse_occlusion_heatmap_llava.png"
    )
