from transformers import pipeline

# Global Initialization of FLAN-T5
style_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-xl",
    tokenizer="google/flan-t5-xl",
    framework="pt"
)

def style_suggestions(input_text, outfit_style, season, occasion):
    """
    Generate fashion advice with personalization for outfit style and season.

    Args:
        input_text (str): User-provided text description or clothing caption.
        outfit_style (str): Selected outfit style (e.g., Casual, Formal).
        season (str): Selected season (e.g., Spring, Summer, Fall, Winter).
        occasion (str): Selected occasion.

    Returns:
        str: Generated styling suggestions.
    """
    context_parts = []  # Start with an empty list for context parts

    if outfit_style and outfit_style.lower() != "none":
        context_parts.append(f"{outfit_style.lower()} outfit")
    if season and season.lower() != "none":
        context_parts.append(f"suitable for {season.lower()}")
    if occasion and occasion.lower() != "none":
        context_parts.append(f"tailored for {occasion.lower()}")

    # Combine all context parts into a single description
    context_description = " and ".join(context_parts)  # Join phrases with "and"

    # Construct the full prompt
    prompt = (
        f"You are a professional stylist. "
        f"The user is looking for {context_description or 'a personalized outfit'}. "
        f"Their input is: '{input_text}'. "
        f"Provide actionable fashion advice including clothing pairings, accessories, and color themes. "
        f"Format the response in bullet points for clarity."
    )

    # Generate styling suggestions using FLAN-T5
    response = style_generator(prompt, max_length=300)
    return response[0]["generated_text"]