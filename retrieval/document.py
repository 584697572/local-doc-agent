from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    表示一份完整文档。

    Loader 负责把 txt / md / pdf 转换成 Document。
    """

    document_id: str
    content: str
    filename: str
    file_type: str

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    表示从 Document 中切出来的一小段内容。

    Retrieval 系统真正检索的基本单位。
    """

    chunk_id: str
    document_id: str
    content: str
    filename: str

    page: Optional[int] = None
    section: Optional[str] = None

    start_char: Optional[int] = None
    end_char: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)