import os
import json
import hashlib
import base64
from pathlib import Path
from typing import List, Dict
from chunking.controller import get_controller
from chunking.parser.fastpdf.util import bytes_to_base64
from chunking.parser.fastpdf.util import OCRMode
from chunking.parser import FastPDF
from chunking.base import CType
from src.constants import PDF_DIR, OUTPUT_DIR, IMAGES_DIR, CHUNK_RECORDS_FILE

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _save_image(content: bytes, page: int, suffix: str) -> str:
    """Save image content to disk with a hashed filename to avoid duplicates.
    Returns the relative path to the saved image."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_hash = _hash_bytes(content)
    filename = f"{page}_{img_hash}.{suffix}"
    path = IMAGES_DIR / filename
    if not path.exists():
        with open(path, "wb") as f:
            f.write(content)
    return str(path)


def parse_pdf(pdf_path: Path) -> List[Dict]:
    """Parse a single PDF into structured chunk records."""
    ctrl = get_controller()
    root = ctrl.as_root_chunk(str(pdf_path))

    chunks = FastPDF.run(
        root,
        use_layout_parser=True,
        render_2d_text_paragraph=True,
        extract_image=True,
        extract_table=True,
    )

    records = []
    child_chunks = [c for _, c in chunks[0].walk() if c.ctype != CType.Root]
    for chunk in child_chunks:
        page = chunk.origin.location["page"]
        bbox = chunk.origin.location["bbox"]
        mimetype = getattr(chunk, "mimetype", None)

        text_content = ""
        image_path = None
        if mimetype in {"image/png", "image/jpeg", "image/jpg"}:
            # save image and keep alt text if any
            suffix = mimetype.split("/")[-1]
            image_path = _save_image(chunk.content, page, suffix)
            text_content = chunk.text or "(image)"
        elif chunk.ctype == CType.Header:
            text_content = f"### {chunk.content}"
        else:
            text_content = chunk.content or ""

        if not text_content and not image_path:
            continue # skip empty chunks

        record = {
            "id": f"{pdf_path.name}_{page}_{hashlib.md5(str(bbox).encode()).hexdigest()[:8]}",
            "source_pdf": pdf_path.name,
            "page": page,
            "bbox": bbox,
            "type": str(chunk.ctype),
            "text": text_content,
            "image_path": image_path,
        }
        records.append(record)
    return records

def main():
    PDF_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_records: List[Dict] = []
    for pdf_file in PDF_DIR.glob("*.pdf"):
        print(f"Parsing {pdf_file} ...")
        records = parse_pdf(pdf_file)
        all_records.extend(records)

    print(f"Total chunks parsed: {len(all_records)}")

    with open(CHUNK_RECORDS_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            # convert bbox to float
            rec['bbox'] = [float(x) for x in rec['bbox']]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote chunks to {CHUNK_RECORDS_FILE}")

if __name__ == "__main__":
    main() 