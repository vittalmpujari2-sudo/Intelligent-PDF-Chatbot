"""
pdf_processor.py - Reads a PDF and cuts its text into small chunks.

Two main jobs:
1. extract_pages()      -> get the text of every page (with page numbers)
2. split_into_chunks()  -> cut each page into overlapping chunks

Chunks are made page by page, so every chunk knows its exact page number.
"""

import io
import re
from typing import Dict, List

from pypdf import PdfReader

import config


def extract_pages(file_bytes: bytes) -> List[Dict]:
    """Return a list like [{"page": 1, "text": "..."}, ...] (page numbers start at 1)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not open this PDF. It may be corrupted. ({exc})")

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # works for PDFs that have only an owner password
        except Exception:
            raise ValueError("This PDF is password protected and cannot be read.")

    pages = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()  # tidy up spaces and line breaks
        if text:
            pages.append({"page": index + 1, "text": text})

    if not pages:
        raise ValueError(
            "No readable text found. This looks like a scanned/image-only PDF "
            "(OCR is not supported in this project)."
        )
    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Cut one long string into overlapping pieces, preferring to cut at sentence ends or spaces."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Look for a nice cut point in the last 40% of the chunk
            window_start = start + int(chunk_size * 0.6)
            cut = text.rfind(". ", window_start, end)
            if cut == -1:
                cut = text.rfind(" ", window_start, end)
            if cut != -1 and cut > start:
                end = cut + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = max(end - overlap, start + 1)  # step back a little = overlap

    return chunks


def split_into_chunks(pages: List[Dict], filename: str) -> List[Dict]:
    """Turn pages into chunks: [{"text", "source", "page"}, ...]."""
    chunks = []
    for page in pages:
        for piece in split_text(page["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append({"text": piece, "source": filename, "page": page["page"]})
    return chunks
