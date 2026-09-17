from retrieval.document import Document, DocumentChunk


def test_create_document():
    doc = Document(
        document_id="doc_001",
        content="这是测试文档。",
        filename="test.txt",
        file_type="txt",
    )

    assert doc.document_id == "doc_001"
    assert doc.filename == "test.txt"
    assert doc.content == "这是测试文档。"


def test_create_document_chunk():
    chunk = DocumentChunk(
        chunk_id="doc_001_chunk_000",
        document_id="doc_001",
        content="这是第一段。",
        filename="test.txt",
        start_char=0,
        end_char=6,
    )

    assert chunk.chunk_id == "doc_001_chunk_000"
    assert chunk.document_id == "doc_001"
    assert chunk.filename == "test.txt"