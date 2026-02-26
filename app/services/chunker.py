"""
Simple text chunker with overlap.
Works for plain text extracted from PDFs or raw strings.
"""
from app.core.config import settings


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text:       Input string to chunk.
        chunk_size: Max words per chunk (defaults to settings.CHUNK_SIZE).
        overlap:    Word overlap between consecutive chunks (defaults to settings.CHUNK_OVERLAP).

    Returns:
        List of text chunk
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide window with overlap

    return [c for c in chunks if c.strip()]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file (bytes)."""
    import io
    import PyPDF2

    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
