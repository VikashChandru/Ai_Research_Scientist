"""
Extract and chunk text from user-uploaded PDFs (or plain-text/CSV files).
"""
import io
from pypdf import PdfReader


def extract_text(file_bytes: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    else:
        # csv / txt / md -> decode as text
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""


def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 30) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks