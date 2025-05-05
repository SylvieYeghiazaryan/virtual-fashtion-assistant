from transformers import pipeline

style_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-xl",  # Using FLAN-T5 model
    tokenizer="google/flan-t5-xl",
    framework="pt"  # Use PyTorch backend
)


def style_suggestions(input_text):
    """
    Generate fashion advice and styling recommendations using FLAN-T5 with optimized prompts.
    Args:
        input_text (str): Text or image-generated caption describing the clothing item/style.
    Returns:
        str: Fashion styling suggestions, formatted as actionable advice.
    """
    prompt = (
        f"You are a professional fashion stylist. "
        f"The user is asking for recommendations for their outfit or style. "
        f"Provide detailed and actionable styling suggestions based on the given description: '{input_text}'. "
        f"Consider the occasion, type of clothing, and color preferences. "
        f"Format the response in bullet points for clarity."
    )

    # Generate response using FLAN-T5
    response = style_generator(prompt, max_length=300)

    return response[0]["generated_text"]