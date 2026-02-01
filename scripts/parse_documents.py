#!/usr/bin/env python3
"""
Parse a folder of PDFs/TXT into structured JSONL.

Features:
- Text extraction from normal PDFs (fast)
- OCR fallback for scanned pages (slower)
- Basic entity pulls (dates/emails/phones) with regex
- Chunking for downstream search/LLM use
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# PDF text extraction
import pdfplumber
from pypdf import PdfReader

# OCR fallback
import pytesseract


# --- Regular expressions (patterns) to find things inside text ---

DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

WHITESPACE_RE = re.compile(r"[ \t]+\n|[ \t]{2,}")


# --- Small helper functions ---

def sha1_of_file(path: Path) -> str:
    """Compute a SHA1 hash of the file bytes so we can uniquely ID the document."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(s: str) -> str:
    """Clean up newlines and extra spaces so text is more consistent."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WHITESPACE_RE.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, max_chars: int = 2500, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split long text into smaller overlapping chunks so it’s easier to search and process later.
    """
    chunks: List[Dict[str, Any]] = []
    if not text:
        return chunks

    i = 0
    chunk_id = 0
    n = len(text)

    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]
        chunks.append({"chunk_id": chunk_id, "text": chunk, "start": i, "end": end})
        chunk_id += 1

        i = end - overlap
        if i < 0:
            i = 0
        if end == n:
            break

    return chunks


# --- PDF extraction functions ---

def extract_pdf_text_fast(path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Try to extract text from a normal (non-scanned) PDF using pdfplumber.
    Also attempt to read basic metadata using pypdf.
    """
    meta: Dict[str, Any] = {"title": None, "author": None}
    text_parts: List[str] = []

    # Read metadata (optional)
    reader = PdfReader(str(path))
    if reader.metadata:
        meta["title"] = getattr(reader.metadata, "title", None)
        meta["author"] = getattr(reader.metadata, "author", None)

    # Extract text page-by-page
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t:
                text_parts.append(t)

    return normalize_text("\n\n".join(text_parts)), meta


def looks_scanned(text: str, min_chars: int = 200) -> bool:
    """
    Heuristic (a smart guess):
    If extracted text is very short, the PDF might be scanned images, not real text.
    """
    return len(text) < min_chars


def ocr_pdf(path: Path, max_pages: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """
    OCR the PDF: turn each page into an image, then read the text from the image.
    This is slower, but works on scanned documents.
    """
    text_parts: List[str] = []
    pages_done = 0

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            if max_pages is not None and pages_done >= max_pages:
                break

            # Render page to an image (PIL Image)
            im = page.to_image(resolution=300).original

            # OCR to get text from the image
            t = pytesseract.image_to_string(im)
            t = t.strip()
            if t:
                text_parts.append(t)

            pages_done += 1

    meta = {"title": None, "author": None}
    return normalize_text("\n\n".join(text_parts)), meta


# --- Entity extraction (simple regex-based) ---

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Pull out dates, emails, phones (very basic) using regex patterns."""
    dates = sorted(set(DATE_RE.findall(text)))
    emails = sorted(set(EMAIL_RE.findall(text)))
    phones = sorted(set(PHONE_RE.findall(text)))
    return {"dates": dates, "emails": emails, "phones": phones}


# --- File discovery ---

def iter_input_files(root: Path) -> List[Path]:
    """Find all .pdf and .txt files under the input folder."""
    exts = {".pdf", ".txt"}
    files: List[Path] = []

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    return sorted(files)


# --- Parse one document into one JSON record ---

def parse_one(path: Path, ocr_if_needed: bool, ocr_max_pages: Optional[int]) -> Dict[str, Any]:
    doc_id = sha1_of_file(path)
    ext = path.suffix.lower()

    record: Dict[str, Any] = {
        "doc_id": doc_id,
        "path": str(path),
        "type": "pdf" if ext == ".pdf" else "txt",
        "pages": None,
        "extraction": {"method": None, "confidence": 0.0},
        "meta": {},
        "text": "",
        "chunks": [],
        "entities": {},
    }

    if ext == ".txt":
        text = normalize_text(path.read_text(errors="ignore"))
        record["extraction"]["method"] = "plain"
        record["text"] = text

    else:
        # PDF
        try:
            text, meta = extract_pdf_text_fast(path)
            record["meta"] = meta

            # Count pages (optional)
            try:
                record["pages"] = len(PdfReader(str(path)).pages)
            except Exception:
                record["pages"] = None

            # If it looks scanned, maybe OCR
            if ocr_if_needed and looks_scanned(text):
                ocr_text, _ = ocr_pdf(path, max_pages=ocr_max_pages)

                # Use OCR text only if it gives more content
                if len(ocr_text) > len(text):
                    text = ocr_text
                    record["extraction"]["method"] = "ocr"
                else:
                    record["extraction"]["method"] = "pypdf/pdfplumber"
            else:
                record["extraction"]["method"] = "pypdf/pdfplumber"

            record["text"] = text

        except Exception as e:
            record["extraction"]["method"] = "error"
            record["meta"] = {"error": repr(e)}
            record["text"] = ""

    record["chunks"] = chunk_text(record["text"])
    record["entities"] = extract_entities(record["text"])
    return record


# --- Command line interface (CLI) entry point ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder containing PDFs/TXTs")
    ap.add_argument("--out", dest="out_file", required=True, help="Output JSONL path")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    ap.add_argument("--ocr-max-pages", type=int, default=None, help="Limit OCR pages per PDF (debug/speed)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_file = Path(args.out_file).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(in_dir)
    if not files:
        raise SystemExit(f"No .pdf or .txt files found under: {in_dir}")

    with out_file.open("w", encoding="utf-8") as f:
        for path in tqdm(files, desc="Parsing"):
            rec = parse_one(path, ocr_if_needed=args.ocr, ocr_max_pages=args.ocr_max_pages)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(files)} docs to {out_file}")


if __name__ == "__main__":
    main()
