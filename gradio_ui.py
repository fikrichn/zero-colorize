import spaces
import os
import torch
import PIL
import gradio as gr

from typing import Optional
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
)
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download


# ========== Initialization ==========

# Ensure required directories exist
os.makedirs("sdxl_light_caption_output", exist_ok=True)

# Download controlnet model snapshot
snapshot_download(
    repo_id='nickpai/sdxl_light_caption_output',
    local_dir='sdxl_light_caption_output'
)

# Device and precision setup
accelerator = Accelerator(mixed_precision="fp16")
weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
device = accelerator.device

print(f"[INFO] Accelerator device: {device}")

# ========== Models ==========

# Pretrained paths
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
safetensors_ckpt = "sdxl_lightning_8step_unet.safetensors"
controlnet_path = "sdxl_light_caption_output/checkpoint-30000/controlnet"

# Load diffusion components
vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
unet = UNet2DConditionModel.from_config(base_model_path, subfolder="unet")
unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", safetensors_ckpt)))

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=weight_dtype)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, vae=vae, unet=unet, controlnet=controlnet
)
pipe.to(device, dtype=weight_dtype)
pipe.safety_checker = None

# Load BLIP captioning model
caption_model_name = "blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(f"Salesforce/{caption_model_name}")
caption_model = BlipForConditionalGeneration.from_pretrained(
    f"Salesforce/{caption_model_name}", torch_dtype=weight_dtype
).to(device)

# ========== Utility Functions ==========

def apply_color(image: PIL.Image.Image, color_map: PIL.Image.Image) -> PIL.Image.Image:
    # Convert to LAB color space
    image_lab = image.convert('LAB')
    color_map_lab = color_map.convert('LAB')

    # Extract and merge LAB channels
    l, _, _ = image_lab.split()
    _, a_map, b_map = color_map_lab.split()
    merged_lab = PIL.Image.merge('LAB', (l, a_map, b_map))

    return merged_lab.convert('RGB')


def remove_unlikely_words(prompt: str) -> str:
    """Removes predefined unlikely phrases from prompt text."""
    unlikely_words = []

    a1 = [f'{i}s' for i in range(1900, 2000)]
    a2 = [f'{i}' for i in range(1900, 2000)]
    a3 = [f'year {i}' for i in range(1900, 2000)]
    a4 = [f'circa {i}' for i in range(1900, 2000)]

    b1 = [f"{y[0]} {y[1]} {y[2]} {y[3]} s" for y in a1]
    b2 = [f"{y[0]} {y[1]} {y[2]} {y[3]}" for y in a1]
    b3 = [f"year {y[0]} {y[1]} {y[2]} {y[3]}" for y in a1]
    b4 = [f"circa {y[0]} {y[1]} {y[2]} {y[3]}" for y in a1]

    manual = [  # same list as your original words_list
        "black and white,", "black and white", "black & white,", "black & white", "circa", 
        "balck and white,", "monochrome,", "black-and-white,", "black-and-white photography,", 
        "black - and - white photography,", "monochrome bw,", "black white,", "black an white,",
        "grainy footage,", "grainy footage", "grainy photo,", "grainy photo", "b&w photo",
        "back and white", "back and white,", "monochrome contrast", "monochrome", "grainy",
        "grainy photograph,", "grainy photograph", "low contrast,", "low contrast", "b & w",
        "grainy black-and-white photo,", "bw", "bw,", "grainy black-and-white photo",
        "b & w,", "b&w,", "b&w!,", "b&w", "black - and - white,", "bw photo,", "grainy  photo,",
        "black-and-white photo,", "black-and-white photo", "black - and - white photography",
        "b&w photo,", "monochromatic photo,", "grainy monochrome photo,", "monochromatic",
        "blurry photo,", "blurry,", "blurry photography,", "monochromatic photo",
        "black - and - white photograph,", "black - and - white photograph", "black on white,",
        "black on white", "black-and-white", "historical image,", "historical picture,", 
        "historical photo,", "historical photograph,", "archival photo,", "taken in the early",
        "taken in the late", "taken in the", "historic photograph,", "restored,", "restored", 
        "historical photo", "historical setting,",
        "historic photo,", "historic", "desaturated!!,", "desaturated!,", "desaturated,", "desaturated", 
        "taken in", "shot on leica", "shot on leica sl2", "sl2",
        "taken with a leica camera", "leica sl2", "leica", "setting", 
        "overcast day", "overcast weather", "slight overcast", "overcast", 
        "picture taken in", "photo taken in", 
        ", photo", ",  photo", ",   photo", ",    photo", ", photograph",
        ",,", ",,,", ",,,,", " ,", "  ,", "   ,", "    ,", 
    ]

    unlikely_words.extend(a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + manual)

    for word in unlikely_words:
        prompt = prompt.replace(word, "")
    return prompt


def get_image_paths(folder_path: str) -> list:
    return [[os.path.join(folder_path, f)] for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png"))]


@spaces.GPU
def process_image(image_path: str,                 
                  positive_prompt: Optional[str],
                  negative_prompt: Optional[str],
                  seed: int) -> tuple[PIL.Image.Image, str]:
    
    """Colorize a grayscale or low-color image using automatic captioning and text-guided diffusion.

    This function performs image-to-image generation using a ControlNet model and Stable Diffusion XL,
    guided by a text caption extracted from the image itself using a BLIP captioning model. Optional
    prompts (positive and negative) can further influence the output style or content.

    Process Overview:
        1. The input image is loaded and resized to 512x512 for inference.
        2. A BLIP model generates a caption describing the image content.
        3. The caption is cleaned using a filtering function to remove misleading or unwanted terms.
        4. A prompt is constructed by combining the user-provided positive prompt with the caption.
        5. A ControlNet-guided image is generated using the SDXL pipeline.
        6. The output image's color channels (A and B in LAB space) are applied to the original luminance (L)
           of the control image to preserve structure while transferring color.
        7. The image is resized back to the original resolution and returned.

    Args:
        image_path: Path to the grayscale or lightly colored input image (JPEG/PNG).
        positive_prompt: Additional descriptive text to enhance or guide the generation.
        negative_prompt: Words or phrases to avoid during generation (e.g., "blurry", "monochrome").
        seed: Random seed for reproducible generation.

    Returns:
        A tuple containing:
            - A colorized PIL image based on the input and generated caption.
            - The cleaned caption string used to guide the generation.
    """

    torch.manual_seed(seed)
    image = PIL.Image.open(image_path)
    original_size = image.size
    control_image = image.convert("L").convert("RGB").resize((512, 512))

    # Image captioning
    input_text = "a photography of"
    inputs = processor(image, input_text, return_tensors="pt").to(device, dtype=weight_dtype)
    caption_ids = caption_model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    caption = remove_unlikely_words(caption)

    # Inference
    final_prompt = [f"{positive_prompt}, {caption}"]
    result = pipe(prompt=final_prompt,
                  negative_prompt=negative_prompt,
                  num_inference_steps=8,
                  generator=torch.manual_seed(seed),
                  image=control_image)

    colorized = apply_color(control_image, result.images[0]).resize(original_size)
    return colorized, caption


# ========== Gradio UI ==========

def create_interface():
    examples = get_image_paths("example/legacy_images")

    return gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(label="Upload Image", type='filepath',
                     value="example/legacy_images/Hollywood-Sign.jpg"),
            gr.Textbox(label="Positive Prompt", placeholder="Enter details to enhance the caption"),
            gr.Textbox(label="Negative Prompt", value="low quality, bad quality, low contrast, black and white, bw, monochrome, grainy, blurry, historical, restored, desaturate"),
        ],
        outputs=[
            gr.Image(label="Colorized Image", format="jpeg",
                     value="example/UUColor_results/Hollywood-Sign.jpeg"),
            gr.Textbox(label="Caption")
        ],
        examples=examples,
        additional_inputs=[gr.Slider(0, 1000, 123, label="Seed")],
        title="Text-Guided Image Colorization",
        description="Upload a grayscale image and generate a color version guided by automatic captioning.",
        cache_examples=False
    )


def main():
    interface = create_interface()
    interface.launch(ssr_mode=False, mcp_server=True)


if __name__ == "__main__":
    main()
