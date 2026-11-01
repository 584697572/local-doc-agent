"""定义完整文档和文档片段这两种数据结构。"""

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

    # 未传入 metadata 时，每个实例会创建自己的空字典，不与其他文档共用。
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

    # 字符范围采用 [start_char, end_char)，包含起点、不包含终点。
    # 外部数据集可能只提供片段正文，此时允许不知道它在原文中的位置。
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # 片段也有独立的补充信息字典，例如保存来源文件类型。
    metadata: Dict[str, Any] = field(default_factory=dict)
