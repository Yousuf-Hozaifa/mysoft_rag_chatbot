import os

# This forces the heavy AI models to download to your E drive instead of C!
os.environ["HF_HOME"] = "E:/huggingface_cache"

import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuration ---
PDF_PATH = "data/Mysoftheaven-Profile 2026.pdf"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def extract_text_from_pdf(file_path: str) -> str:
    """Reads the PDF file and extracts all text."""
    print(f"Reading PDF: {file_path}")
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"⚠️ Could not find the PDF at {file_path}. Make sure it is in the 'data' folder!")

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Splits the giant text wall into smaller, searchable paragraphs (chunks)."""
    print(f"Chunking text (Size: {chunk_size}, Overlap: {overlap})...")
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        
    print(f"Created {len(chunks)} chunks.")
    return chunks

def build_and_save_database(chunks: list):
    """Converts text chunks to vectors and saves them to a FAISS database."""
    print("Loading embedding model (this may take a moment)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Converting text chunks into vector embeddings...")
    embeddings = embedder.encode(chunks).astype("float32")
    
    print("Building FAISS Vector Database...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    np.save(f"{INDEX_DIR}/chunks.npy", np.array(chunks))
    
    print(f"Database saved successfully in the '{INDEX_DIR}' folder!")

if __name__ == "__main__":
    print("Starting Data Ingestion Pipeline...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    text_chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    build_and_save_database(text_chunks)
    print("Ingestion complete! You are ready for the next step.")