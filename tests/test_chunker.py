import pytest

from retrieval.document import Document
from retrieval.chunker import chunk_document


def test_chunk_short_document():
    doc = Document(
        document_id="doc_001",
        content="这是一个很短的测试文档。",
        filename="demo.txt",
        file_type="txt",
    )

    chunks = chunk_document(
        doc,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1

    assert chunks[0].chunk_id == "doc_001_chunk_0000"
    assert chunks[0].document_id == "doc_001"
    assert chunks[0].content == "这是一个很短的测试文档。"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(doc.content)

def test_chunk_document_with_overlap():
    content = "abcdefghij"

    doc = Document(
        document_id="doc_002",
        content=content,
        filename="demo.txt",
        file_type="txt",
    )

    chunks = chunk_document(
        doc,
        chunk_size=5,
        overlap=2,
    )

    assert len(chunks) == 3

    assert chunks[0].content == "abcde"
    assert chunks[1].content == "defgh"
    assert chunks[2].content == "ghij"

def test_invalid_chunk_size():
    doc = Document(
        document_id="doc_003",
        content="测试内容",
        filename="demo.txt",
        file_type="txt",
    )

    with pytest.raises(ValueError):
        chunk_document(
            doc,
            chunk_size=0,
            overlap=0,
        )

def test_invalid_overlap():
    doc = Document(
        document_id="doc_004",
        content="测试内容",
        filename="demo.txt",
        file_type="txt",
    )

    with pytest.raises(ValueError):
        chunk_document(
            doc,
            chunk_size=100,
            overlap=100,
        )

def test_pdf_chunk_page_number():
    content = "AAAAA\n\nBBBBB"

    doc = Document(
        document_id="paper",
        content=content,
        filename="paper.pdf",
        file_type="pdf",
        metadata={
            "page_boundaries": [
                {
                    "page": 1,
                    "start_char": 0,
                    "end_char": 5,
                },
                {
                    "page": 2,
                    "start_char": 7,
                    "end_char": 12,
                },
            ]
        },
    )

    chunks = chunk_document(
        doc,
        chunk_size=5,
        overlap=0,
    )

    assert chunks[0].page == 1