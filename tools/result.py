"""统一的工具执行结果。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    Tool 执行后的统一返回结构。

    status:
        success / empty / error
    """

    status: str
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, metadata: dict[str, Any] | None = None):
        """工具成功，并返回有效内容。"""
        if not metadata:
            metadata = {}

        return cls(status="success", content=content, metadata=metadata)

    @classmethod
    def empty(
        cls,
        content: str = "工具执行成功，但没有找到结果。",
        metadata: dict[str, Any] | None = None,
    ):
        """工具正常执行，但没有有效结果。"""
        if not metadata:
            metadata = {}

        return cls(status="empty", content=content, metadata=metadata)

    @classmethod
    def failure(cls, error: str, metadata: dict[str, Any] | None = None):
        """工具执行失败。"""
        if not metadata:
            metadata = {}

        return cls(status="error", error=error, metadata=metadata)

    def to_message_content(self) -> str:
        """
        转成可以发送给 LLM 的文本。
        """

        if self.status == "success":
            return self.content

        if self.status == "empty":
            return self.content

        error_message = self.error
        if not error_message:
            error_message = "未知错误"
        return f"工具执行失败：{error_message}"
