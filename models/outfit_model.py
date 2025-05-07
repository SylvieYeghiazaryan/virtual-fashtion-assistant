from diffusers import StableDiffusionPipeline

device = "cpu"
stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    low_cpu_mem_usage=True
).to(device)


def generate_outfit(prompt):
    """
    Generate an outfit image based on user styling suggestions using Stable Diffusion.

    Args:
        prompt (str): Text prompt describing the outfit/style.

    Returns:
        PIL.Image.Image: Generated outfit image.
    """
    print(f"Generating outfit with prompt: {prompt}")
    image = stable_diffusion_pipe(prompt).images[0]
    print("Outfit image generated successfully!")
    return image