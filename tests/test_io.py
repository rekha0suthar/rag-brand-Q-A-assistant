from utils.io import clean_text, has_enough_text, MIN_DOC_CHARS


def test_collapses_internal_whitespace():
    assert clean_text("hello   world") == "hello world"


def test_collapses_newlines_and_tabs():
    assert clean_text("a\n\nb\tc\r\nd") == "a b c d"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("  padded  ") == "padded"


def test_empty_and_whitespace_only():
    assert clean_text("") == ""
    assert clean_text(" \n\t ") == ""


def test_plain_text_unchanged():
    assert clean_text("already clean") == "already clean"


# ---- low-text guard (scanned/image PDFs yield near-empty extraction) ----

def test_has_enough_text_rejects_empty_and_none():
    assert not has_enough_text("")
    assert not has_enough_text(None)
    assert not has_enough_text("   \n\t  ")


def test_has_enough_text_rejects_scanned_pdf_remnants():
    # pypdf on a scanned PDF typically yields a few stray characters.
    assert not has_enough_text("Page 1\x0cPage 2")


def test_has_enough_text_boundary():
    assert not has_enough_text("x" * (MIN_DOC_CHARS - 1))
    assert has_enough_text("x" * MIN_DOC_CHARS)


def test_has_enough_text_custom_min():
    assert has_enough_text("short but fine", min_chars=5)
