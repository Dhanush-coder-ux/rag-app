from app.core.config import settings
import io
import PyPDF2

def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:

    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  

    return [c for c in chunks if c.strip()]


def extract_text_from_pdf(file_bytes: bytes) -> str:

    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
