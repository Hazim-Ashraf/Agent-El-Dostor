"""Contract ingestion pipeline (M1).

parse (PDF / DOCX / TXT) -> OCR fallback (scanned) -> clause segmentation ->
embed -> store. Exposes `ingest_contract_bytes` for the GUI and a CLI for files.

Run inside the container:
    docker compose run --rm app python -m app.ingestion.contracts \
        --file data/contracts/sample_employment_contract_en.txt --type employment
"""
from __future__ import annotations

import argparse
import os
import re
import uuid
from typing import Any

from app.core.logging import get_logger
from app.retrieval import embeddings, store

log = get_logger(__name__)

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")

# Clause headers we try to recognise (English + Arabic).
_REF_PATTERNS = [
    re.compile(r"^\s*(article|clause|section)\s+([0-9٠-٩]+)", re.IGNORECASE),
    re.compile(r"^\s*(المادة|البند|الفصل)\s*([0-9٠-٩]+)"),
    re.compile(r"^\s*([0-9٠-٩]+)\s*[.)\-:]"),
]


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #


def _extract_pdf(data: bytes) -> str:
    import io

    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n\n".join(pages).strip()

    # If a digital extraction yielded almost nothing, treat it as scanned -> OCR.
    if len(text) < 100:
        log.info("PDF has little embedded text (%s chars) — running OCR.", len(text))
        text = _ocr_pdf(data)
    return text


def _ocr_pdf(data: bytes) -> str:
    from pdf2image import convert_from_bytes

    import pytesseract

    images = convert_from_bytes(data, dpi=200)
    return "\n\n".join(
        pytesseract.image_to_string(img, lang="ara+eng") for img in images
    ).strip()


def _extract_image(data: bytes) -> str:
    import io

    from PIL import Image
    import pytesseract

    image = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(image, lang="ara+eng").strip()


def _extract_docx(data: bytes) -> str:
    import io

    import docx

    document = docx.Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()


def extract_text(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
        return _extract_image(data)
    # .txt, .md, or unknown -> decode as text
    return data.decode("utf-8", errors="replace").strip()


# --------------------------------------------------------------------------- #
# Language + segmentation
# --------------------------------------------------------------------------- #


def detect_lang(text: str) -> str:
    ar = len(_ARABIC.findall(text))
    en = len(_LATIN.findall(text))
    if ar and en:
        return "mixed" if min(ar, en) / max(ar, en) > 0.2 else ("ar" if ar > en else "en")
    if ar:
        return "ar"
    return "en"


def _detect_ref(block: str) -> str | None:
    first_line = block.strip().splitlines()[0] if block.strip() else ""
    for pattern in _REF_PATTERNS:
        m = pattern.match(first_line)
        if m:
            return m.group(0).strip()
    return None


def _split_long(block: str, max_chars: int = 1200) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in block.splitlines(keepends=True):
        if size + len(line) > max_chars and current:
            pieces.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        pieces.append("".join(current))
    return pieces


def segment_clauses(text: str) -> list[dict[str, Any]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    clauses: list[dict[str, Any]] = []
    index = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        for piece in _split_long(block):
            piece = piece.strip()
            if len(piece) < 3:
                continue
            index += 1
            clauses.append(
                {
                    "clause_index": index,
                    "clause_ref": _detect_ref(piece),
                    "lang": detect_lang(piece),
                    "text": piece,
                }
            )
    return clauses


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def ingest_contract_bytes(
    filename: str,
    data: bytes,
    contract_type: str = "unknown",
    conn=None,
) -> dict[str, Any]:
    """Parse + segment + embed + store one contract. Returns metadata."""
    text = extract_text(filename, data)
    if not text.strip():
        raise ValueError("No text could be extracted from the file.")

    clauses = segment_clauses(text)
    if not clauses:
        raise ValueError("The document produced no usable clauses.")

    own_conn = conn is None
    if own_conn:
        conn = store.connect()
    try:
        dim = embeddings.dimension()
        store.init_contract_schema(conn, dim)

        contract_id = f"c_{uuid.uuid4().hex[:12]}"
        vectors = embeddings.embed_passages([c["text"] for c in clauses])
        rows = [
            {
                "id": f"{contract_id}::{c['clause_index']}",
                "contract_id": contract_id,
                "clause_index": c["clause_index"],
                "clause_ref": c["clause_ref"],
                "lang": c["lang"],
                "text": c["text"],
                "embedding": vector,
            }
            for c, vector in zip(clauses, vectors)
        ]

        contract_lang = detect_lang(text)
        store.upsert_contract(
            conn,
            {
                "id": contract_id,
                "filename": filename,
                "contract_type": contract_type,
                "lang": contract_lang,
                "n_clauses": len(rows),
            },
        )
        store.replace_contract_clauses(conn, contract_id, rows)
        log.info("Ingested contract %s (%s clauses) from %s", contract_id, len(rows), filename)
        return {
            "contract_id": contract_id,
            "filename": filename,
            "contract_type": contract_type,
            "lang": contract_lang,
            "n_clauses": len(rows),
        }
    finally:
        if own_conn:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a contract file.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--type", default="unknown", help="employment | rental | shareholder | service | commercial | unknown")
    args = parser.parse_args()

    with open(args.file, "rb") as fh:
        data = fh.read()
    meta = ingest_contract_bytes(os.path.basename(args.file), data, args.type)
    print(meta)


if __name__ == "__main__":
    main()
