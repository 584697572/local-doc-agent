from pathlib import Path

from retrieval.document import Document


def _read_text_file(path: Path) -> str:
    """
    读取文本文件。

    优先使用 utf-8；
    如果失败，再尝试 gbk。
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk")


def load_txt_file(file_path: str | Path) -> Document:
    """
    把 txt 文件加载为 Document。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"不是 txt 文件: {path}")

    content = _read_text_file(path)

    return Document(
        document_id=path.stem,
        content=content,
        filename=path.name,
        file_type="txt",
        metadata={
            "source_path": str(path),
        },
    )


def load_markdown_file(file_path: str | Path) -> Document:
    """
    把 Markdown 文件加载为 Document。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.suffix.lower() != ".md":
        raise ValueError(f"不是 Markdown 文件: {path}")

    content = _read_text_file(path)

    return Document(
        document_id=path.stem,
        content=content,
        filename=path.name,
        file_type="md",
        metadata={
            "source_path": str(path),
        },
    )


def load_document(file_path: str | Path) -> Document:
    """
    文档加载统一入口。

    根据文件扩展名，自动选择对应 Loader。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt_file(path)

    if suffix == ".md":
        return load_markdown_file(path)

    raise ValueError(f"暂不支持的文件类型: {suffix}")