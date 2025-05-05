from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def image_caption(image_path):
    """
    Generate a caption for the uploaded clothing image using preloaded BLIP model.
    Args:
        image_path (str): Path to the uploaded image.
    Returns:
        str: Caption describing the uploaded image.
    """
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption