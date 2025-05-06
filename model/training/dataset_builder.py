import os
import json
from .pdf_extractor import extract_text_from_pdf
from .preprocess import extract_abstract_and_body, preprocess_text
from .config import PDF_FOLDER, DATASET_PATH

def build_dataset():
    """
    Processes all PDFs in the PDF_FOLDER and creates a JSONL dataset
    with 'text' (body) and 'summary' (abstract) fields.
    """
    dataset = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            full_path = os.path.join(PDF_FOLDER, filename)
            raw_text = extract_text_from_pdf(full_path)
            abstract, body = extract_abstract_and_body(raw_text)
            body = preprocess_text(body)
            abstract = preprocess_text(abstract)
            if len(body) > 500 and len(abstract) > 20:
                dataset.append({"text": body, "summary": abstract})

    with open(DATASET_PATH, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset built with {len(dataset)} samples at {DATASET_PATH}")

if __name__ == "__main__":
    build_dataset()
