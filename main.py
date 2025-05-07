import gradio as gr
from PIL import Image
from models.caption_model import image_caption
from models.style_model import style_suggestions
from models.outfit_model import generate_outfit
from models.variation_model import generate_variations
from models.translation_model import load_translation_model, translate_text

# Translation models (English used as intermediary language)
translation_models = {}
translation_models["to_english"] = {}
translation_models["from_english"] = {}

# Preload frequently used translation models
for lang in ["en", "es", "fr", "de"]:  # English, Spanish, French, German
    if lang != "en":
        translation_models["to_english"][lang] = load_translation_model(lang, "en")
        translation_models["from_english"][lang] = load_translation_model("en", lang)


def validate_inputs(text_input, image_input):
    """Ensure user inputs are valid."""
    if not text_input.strip():
        raise ValueError("Text input cannot be empty. Please describe your style or occasion.")
    if image_input and not isinstance(image_input, Image.Image):
        raise ValueError("Uploaded file is not a valid image.")
    return True


def fashion_pipeline(text_input, image_input, outfit_style, season, occasion, toggle_variations, selected_language,
                     num_variations=3):
    """
    Generate styling suggestions and multiple outfit variations.

    Args:
        text_input (str): Description of style/occasion.
        image_input (PIL.Image or None): Optional uploaded image for contextual styling.
        outfit_style (str): Desired outfit style (e.g., Casual, Formal).
        season (str): Desired season (e.g., Spring, Summer).
        occasion (str): Desired occasion (e.g., Business Meeting, None).
        toggle_variations (bool): Whether to use image-to-image transformations for visuals.
        selected_language (str): User's preferred language.
        num_variations (int): Number of variations to generate for visuals.

    Returns:
        tuple: Caption for uploaded image, single styling suggestion, multiple generated visuals.
    """
    # Initialize result placeholders
    results = {"image_caption": "", "styling_advice": "", "generated_images": []}
    english_input_text = text_input
    to_english_model, to_english_tokenizer = translation_models["to_english"].get(selected_language, (None, None))
    from_english_model, from_english_tokenizer = translation_models["from_english"].get(selected_language, (None, None))

    # Translate input text to English if necessary
    if selected_language != "en" and to_english_model and to_english_tokenizer:
        try:
            english_input_text = translate_text(text_input, to_english_model, to_english_tokenizer)
        except Exception as e:
            return f"Error translating input text: {str(e)}", "", None

    validate_inputs(english_input_text, image_input)

    # Step 1: Captioning Uploaded Image
    if image_input:
        try:
            image_input.save("temp_clothing_image.jpg")
            results["image_caption"] = image_caption("temp_clothing_image.jpg")
        except Exception as e:
            results["image_caption"] = f"Error generating caption: {str(e)}"

    # Combine image caption and text input
    combined_input = (english_input_text + " " + results["image_caption"]).strip()

    # Step 2: Generate a Single Styling Suggestion (Occasion-aware)
    try:

        max_prompt_length = 175
        styling_advice = style_suggestions(combined_input, outfit_style, season, occasion)[:max_prompt_length]

        # Translate styling suggestions back into the user's preferred language
        if selected_language != "en" and from_english_model and from_english_tokenizer:
            styling_advice = translate_text(styling_advice, from_english_model, from_english_tokenizer)

        results["styling_advice"] = styling_advice
    except Exception as e:
        results["styling_advice"] = f"Error generating styling advice: {str(e)}"

    # Step 3: Generate Multiple Visual Outputs (Stable Diffusion)
    try:
        for _ in range(num_variations):  # Generate `num_variations` images
            if image_input and toggle_variations:
                generated_image = generate_variations("temp_clothing_image.jpg", results["styling_advice"])
            else:
                generated_image = generate_outfit(results["styling_advice"])

            results["generated_images"].append(generated_image)
    except Exception as e:
        results["generated_images"].append(f"Error generating outfit image: {str(e)}")

    return results["image_caption"], results["styling_advice"], results["generated_images"]

def gradio_interface():
    """Launch the Gradio interface."""
    language_selector = gr.Dropdown(
        choices=["en", "es", "fr", "de"], label="Select Your Language", value="en"
    )
    outfit_style = gr.Dropdown(
        choices=["None", "Casual", "Formal", "Streetwear", "Boho", "Business Casual"],
        label="Select Desired Outfit Style",
        value="Casual"
    )
    season = gr.Dropdown(
        choices=["None", "Spring", "Summer", "Fall", "Winter"],
        label="Select Desired Season",
        value="Spring"
    )
    occasion = gr.Dropdown(
        choices=["None", "Business Meeting", "Date Night", "Wedding", "Gym"],
        label="Select Occasion",
        value="None"
    )
    toggle_variations = gr.Checkbox(label="Use Image-to-Image Variations", value=False)
    num_variations = gr.Slider(label="Number of Outfit Variations", minimum=1, maximum=5, value=3, step=1)
    text_input = gr.Textbox(label="Describe your style or occasion", placeholder="e.g., 'Formal evening wear'")
    image_input = gr.Image(label="Upload a clothing image (optional)", type="pil")
    caption_output = gr.Textbox(label="Generated Caption for Uploaded Image")
    styling_output = gr.Textbox(label="Single Styling Suggestion")
    image_gallery = gr.Gallery(label="Generated Outfit Images").style(grid=[2], height="auto")

    # Gradio interface
    interface = gr.Interface(
        fn=fashion_pipeline,
        inputs=[text_input, image_input, outfit_style, season, occasion, toggle_variations, language_selector, num_variations],
        outputs=[caption_output, styling_output, image_gallery],
        title="Virtual Fashion Assistant with Event-Specific Recommendations and Variations",
        description="Generate a single styling suggestion paired with multiple outfit designs for comparison.",
    )
    interface.launch()

if __name__ == "__main__":
    gradio_interface()