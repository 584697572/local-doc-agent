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