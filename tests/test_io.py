from utils.io import clean_text


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
