from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Configure OpenAI client to use OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "rag-app"),
    }
)
EMBED_MODEL = "openai/text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def load_and_chunk_txt(path: str):
    # Try different encodings to handle various text file formats
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

    text = None
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        # Fallback: read as binary and decode with error handling
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8', errors='replace')

    chunks = splitter.split_text(text)
    return chunks


def load_and_chunk_document(path: str):
    file_path = Path(path)
    file_ext = file_path.suffix.lower()

    if file_ext == '.pdf':
        return load_and_chunk_pdf(path)
    elif file_ext == '.txt':
        return load_and_chunk_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .pdf, .txt")


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]