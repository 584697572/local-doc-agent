from pathlib import Path

from retrieval.document import Document


def load_txt_file(file_path: str | Path) -> Document:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"不是 txt 文件: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gbk")

    return Document(
        document_id=path.stem,
        content=content,
        filename=path.name,
        file_type="txt",
        metadata={
            "source_path": str(path),
        },
    )