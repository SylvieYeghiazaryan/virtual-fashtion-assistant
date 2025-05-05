import gradio as gr
from models.caption_model import image_caption
from models.style_model import style_suggestions
from models.outfit_model import generate_outfit

def fashion_pipeline(text_input, image_input):
    """
    Handles input processing and generates styling suggestions and outfit images.
    Args:
        text_input (str): User-provided text input.
        image_input (PIL.Image): Uploaded clothing image.
    Returns:
        tuple: (Styling suggestions, Generated outfit image)
    """
    caption = ""

    # Step 1: Process uploaded image (if provided)
    if image_input is not None:
        image_input.save("temp_clothing_image.jpg")
        caption = image_caption("temp_clothing_image.jpg")

    # Step 2: Combine text input and caption
    combined_input = (text_input + " " + caption).strip()

    # Step 3: Generate styling suggestions
    styling_advice = style_suggestions(combined_input)

    # Step 4: Generate outfit based on suggestions
    outfit_image = generate_outfit(styling_advice)

    return styling_advice, outfit_image

def gradio_interface():
    """
    Gradio interface for interacting with the fashion assistant.
    """
    text_input = gr.Textbox(label="Describe your style or occasion")
    image_input = gr.Image(label="Upload a clothing image (optional)", type="pil")
    text_output = gr.Textbox(label="Styling Suggestions")
    image_output = gr.Image(label="Generated Outfit Image")

    interface = gr.Interface(
        fn=fashion_pipeline,
        inputs=[text_input, image_input],
        outputs=[text_output, image_output],
        title="AI Virtual Fashion Assistant",
        description="Get outfit ideas and visual suggestions based on your preferences. Powered by AI!",
    )

    interface.launch()

if __name__ == "__main__":
    gradio_interface()