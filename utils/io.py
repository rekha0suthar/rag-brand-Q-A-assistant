from pathlib import Path
from typing import List, Tuple
import re

def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_docs(docs_dir: str = "docs") -> List[Tuple[str, str]]:
    """
    Returns list of (filename, cleaned_text)
    """
    root = Path(docs_dir)
    out = []
    for p in root.glob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            t = load_markdown(p)
        elif p.suffix.lower() == ".pdf":
            t = load_pdf(p)
        else:
            continue
        out.append((p.name, clean_text(t)))
    return out
