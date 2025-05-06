import re

def extract_abstract_and_body(text):
    """
    Extracts the abstract and main body from the research paper text.
    """
    abstract_match = re.search(r'(?i)(abstract)(.*?)(introduction|1\.|I\.)', text, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(2).strip()
        body = text[abstract_match.end():].strip()
    else:
        # Fallback: use first 250 words as abstract
        words = text.split()
        abstract = " ".join(words[:250])
        body = " ".join(words[250:])
    return abstract, body

def preprocess_text(text):
    """
    Cleans the text by removing references, acknowledgments, and extra whitespace.
    """
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.split(r'\nreferences\n|\nREFERENCES\n', text, flags=re.IGNORECASE)[0]
    text = re.split(r'\nacknowledg(e)?ments?\n', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'Figure \d+.*\n', '', text)
    text = re.sub(r'Table \d+.*\n', '', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()
