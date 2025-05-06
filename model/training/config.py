import os

# Path to the folder containing PDFs
PDF_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/pdfs"))

# Path to save the processed dataset
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/arxiv_summarization.jsonl"))

# Model and tokenizer names
MODEL_NAME = "facebook/bart-large-cnn"

# Training parameters
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 1
NUM_EPOCHS = 1
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
