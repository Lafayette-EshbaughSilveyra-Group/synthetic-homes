"""
llava_processing.py

Uses LLaVA to extract descriptive information from property photos and sketches.

Responsibilities:
- Describe exterior features from property photos.
- Describe floorplans and room dimensions from sketches.
- Outputs descriptive text to feed into geometry_generation.py.
"""


from transformers import pipeline
from PIL import Image

# Initialize LLaVA pipeline globally
llava_pipeline = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf", device_map="auto")


def run_llava(image_path: str, prompt: str) -> str:
    """
    Runs LLaVA inference on the provided image with a given prompt.

    Args:
        image_path (str): Path to the image.
        prompt (str): Instruction prompt for the vision model.

    Returns:
        str: LLaVA model's descriptive output.
    """
    image = Image.open(image_path).convert("RGB")
    result = llava_pipeline({
        "image": image,
        "prompt": prompt
    })
    return result[0]['generated_text']


def describe_exterior(image_path: str) -> str:
    """
    Runs LLaVA on the exterior image to generate a descriptive summary.

    Args:
        image_path (str): Path to the exterior photo.

    Returns:
        str: Description of visible exterior features.
    """
    prompt = (
        "Describe the visible exterior features of this residential building in detail. "
        "Mention materials (brick, siding, etc.), number of visible stories, roof type, windows, doors, and any other notable architectural features. "
        "Do not speculate beyond what is clearly visible."
    )
    return run_llava(image_path, prompt)


def describe_floorplan(sketch_path: str) -> str:
    """
    Runs LLaVA on the floorplan sketch to extract room names and approximate dimensions.

    Args:
        sketch_path (str): Path to the sketch image.

    Returns:
        str: Description of labeled rooms and approximate layout.
    """
    prompt = (
        "Extract and list all labeled rooms and their approximate dimensions from this floorplan sketch. "
        "Reproduce the room names and measurements exactly as written. "
        "Do not summarize or invent details beyond what is explicitly labeled."
    )
    return run_llava(sketch_path, prompt)
