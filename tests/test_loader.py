import pytest

from retrieval.loader import load_txt_file


def test_load_txt_file(tmp_path):
    file_path = tmp_path / "demo.txt"
    file_path.write_text("这是一个测试文档。", encoding="utf-8")

    doc = load_txt_file(file_path)

    assert doc.document_id == "demo"
    assert doc.filename == "demo.txt"
    assert doc.file_type == "txt"
    assert doc.content == "这是一个测试文档。"


def test_load_missing_txt_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_txt_file(file_path)

import pytest

from retrieval.loader import (
    load_txt_file,
    load_markdown_file,
    load_document,
)


def test_load_txt_file(tmp_path):
    file_path = tmp_path / "demo.txt"
    file_path.write_text("这是一个测试文档。", encoding="utf-8")

    doc = load_txt_file(file_path)

    assert doc.document_id == "demo"
    assert doc.filename == "demo.txt"
    assert doc.file_type == "txt"
    assert doc.content == "这是一个测试文档。"


def test_load_missing_txt_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_txt_file(file_path)


def test_load_markdown_file(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# RAG\n\n这是 Markdown 测试文档。",
        encoding="utf-8",
    )

    doc = load_markdown_file(file_path)

    assert doc.document_id == "notes"
    assert doc.filename == "notes.md"
    assert doc.file_type == "md"
    assert "# RAG" in doc.content
    assert "这是 Markdown 测试文档。" in doc.content


def test_load_document_txt(tmp_path):
    file_path = tmp_path / "demo.txt"
    file_path.write_text("TXT 内容", encoding="utf-8")

    doc = load_document(file_path)

    assert doc.file_type == "txt"
    assert doc.content == "TXT 内容"


def test_load_document_markdown(tmp_path):
    file_path = tmp_path / "demo.md"
    file_path.write_text("# 标题", encoding="utf-8")

    doc = load_document(file_path)

    assert doc.file_type == "md"
    assert doc.content == "# 标题"


def test_load_document_unsupported_file_type(tmp_path):
    file_path = tmp_path / "demo.docx"
    file_path.write_text("fake docx", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(file_path)
        
from retrieval.loader import (
    load_txt_file,
    load_markdown_file,
    load_pdf_file,
    load_document,
)

def test_load_missing_pdf_file(tmp_path):
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf_file(file_path)