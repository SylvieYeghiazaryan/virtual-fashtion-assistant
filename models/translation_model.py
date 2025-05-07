from transformers import MarianMTModel, MarianTokenizer

def load_translation_model(src_lang, tgt_lang):
    """
    Load MarianMT translation model for given source and target languages.

    Args:
        src_lang (str): Source language (e.g., "en" for English).
        tgt_lang (str): Target language (e.g., "es" for Spanish).

    Returns:
        tuple: MarianMTModel and MarianTokenizer.
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Translate text from one language to another
def translate_text(input_text, model, tokenizer):
    """
    Translate text using the MarianMT model.

    Args:
        input_text (str): Text to be translated.
        model (MarianMTModel): MarianMT translation model.
        tokenizer (MarianTokenizer): Tokenizer for the model.

    Returns:
        str: Translated text.
    """
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text