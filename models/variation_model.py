from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

def generate_variations(image_path, prompt, strength=0.8):
    """
    Generate variations of an uploaded clothing image using Stable Diffusion img2img.
    Args:
        image_path (str): Path to the uploaded image.
        prompt (str): Text input describing the desired style for the variation.
        strength (float): Degree of variation (0.0 to 1.0).
    Returns:
        PIL.Image.Image: Generated variation of the input image.
    """
    init_image = Image.open(image_path).convert("RGB")
    image = img2img_pipe(prompt=prompt, init_image=init_image, strength=strength).images[0]
    return image