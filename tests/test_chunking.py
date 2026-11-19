from ingest import chunk_texts
from utils.io import load_bytes


def test_chunk_texts_tags_source_metadata():
    chunks = chunk_texts([("a.md", "short text"), ("b.txt", "other text")])
    assert [c["metadata"]["source"] for c in chunks] == ["a.md", "b.txt"]
    assert all(c["page_content"].strip() for c in chunks)


def test_chunk_texts_splits_long_text():
    long_text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_texts([("big.md", long_text)], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c["page_content"]) <= 200 for c in chunks)


def test_chunk_texts_skips_empty_docs():
    assert chunk_texts([("empty.md", ""), ("blank.txt", "   ")]) == []


def test_load_bytes_text_formats():
    assert load_bytes("note.md", b"# Title\n\nHello   world") == "# Title Hello world"
    assert load_bytes("note.txt", b"  plain \n text ") == "plain text"


def test_load_bytes_rejects_unknown_extension():
    import pytest

    with pytest.raises(ValueError):
        load_bytes("image.png", b"\x89PNG")
