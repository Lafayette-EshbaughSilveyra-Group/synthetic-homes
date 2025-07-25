"""
llava_processing.py

Uses LLaVA to extract descriptive information from property photos and sketches.

Responsibilities:
- Describe exterior features from property photos.
- Describe floorplans and room dimensions from sketches.
- Outputs descriptive text to feed into geometry_generation.py.
"""
import gc

import torch.cuda
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Initialize LLaVA pipeline globally
model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
   quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()


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
    formatted_prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
    inputs = processor(formatted_prompt, image, return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=200)
    raw_result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    result = raw_result.split("ASSISTANT:")[-1].strip()

    del image
    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        print(f"[INFO] GPU Allocated: {torch.cuda.memory_allocated() / 1e6:2f} MB")
        print(f"[INFO] GPU Reserved: {torch.cuda.memory_reserved() / 1e6:2f} MB")

    return result


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
