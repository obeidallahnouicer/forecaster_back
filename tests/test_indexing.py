import os
from tools import index_md


def test_split_markdown_into_sections_basic():
    md = """# Title\nFirst paragraph.\n## Sub\nMore content here."""
    sections = index_md.split_markdown_into_sections(md)
    assert isinstance(sections, list)
    assert any('First paragraph' in s.get('text','') for s in sections)


def test_chunk_text_overlap_and_size():
    text = "a" * 250  # long string of 250 chars
    chunks = index_md.chunk_text(text, chunk_size=100, overlap=10)
    # should produce multiple chunks
    assert len(chunks) >= 3
    # ensure overlap by checking adjacent chunks share some substring
    assert chunks[0][-10:] == chunks[1][:10]
