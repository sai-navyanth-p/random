from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from .config import OUTPUT_DIR, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH

def load_model_and_tokenizer():
    """
    Loads the fine-tuned BART model and tokenizer from the output directory.
    """
    tokenizer = BartTokenizer.from_pretrained(OUTPUT_DIR)
    model = BartForConditionalGeneration.from_pretrained(OUTPUT_DIR)
    model.eval()
    return tokenizer, model

def summarize(text, tokenizer, model):
    """
    Generates a summary for the given text using the fine-tuned BART model.
    """
    inputs = tokenizer(
        [text],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=MAX_TARGET_LENGTH,
            min_length=30,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    import sys

    tokenizer, model = load_model_and_tokenizer()

    if len(sys.argv) > 1:
        # Summarize text from a file
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()
        summary = summarize(text, tokenizer, model)
        print("Summary:\n", summary)
    else:
        # Summarize a hardcoded or user-input text
        print("Enter/paste the text to summarize (end with Ctrl+D):")
        text = sys.stdin.read()
        summary = summarize(text, tokenizer, model)
        print("Summary:\n", summary)
