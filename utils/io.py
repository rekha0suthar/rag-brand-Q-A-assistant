"""Document loading and text extraction (.md/.txt/.pdf), from disk or bytes."""

import re
from pathlib import Path


def load_text(path: Path) -> str:
    """Read a plain-text file (.md/.txt), ignoring undecodable bytes."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    """Extract the text layer of a PDF (empty string per page without one)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


# Below this many extracted characters a "document" is almost certainly a
# scanned/image PDF (pypdf extracts no text layer) or an empty file --
# indexing it would only produce garbage matches.
MIN_DOC_CHARS = 200


def has_enough_text(text: str, min_chars: int = MIN_DOC_CHARS) -> bool:
    """True when extracted text is long enough to be worth indexing."""
    return len((text or "").strip()) >= min_chars


def load_bytes(filename: str, data: bytes) -> str:
    """
    Extract cleaned text from raw file bytes (.md/.txt/.pdf) -- the in-memory
    counterpart of load_docs, used for Streamlit uploads that never touch disk.
    Raises ValueError for unsupported extensions.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".txt"}:
        text = data.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported file type: {suffix or filename}")
    return clean_text(text)


def load_docs(docs_dir: str = "docs") -> list[tuple[str, str]]:
    """Load every supported file in docs_dir as (filename, cleaned_text) pairs.

    Unsupported extensions are skipped silently -- docs_dir is a curated
    corpus directory, not arbitrary user input.
    """
    root = Path(docs_dir)
    out = []
    for p in root.glob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            t = load_text(p)
        elif p.suffix.lower() == ".pdf":
            t = load_pdf(p)
        else:
            continue
        out.append((p.name, clean_text(t)))
    return out
