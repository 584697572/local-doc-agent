"""ToolResult 测试。"""

from tools.result import ToolResult


def test_success_result():
    result = ToolResult.success(
        content="有效证据",
        metadata={
            "result_count": 1,
        },
    )

    assert result.status == "success"
    assert result.content == "有效证据"
    assert result.error is None
    assert result.metadata["result_count"] == 1


def test_empty_result():
    result = ToolResult.empty(
        content="没有找到结果。"
    )

    assert result.status == "empty"
    assert result.to_message_content() == "没有找到结果。"


def test_error_result():
    result = ToolResult.failure(
        error="模型加载失败"
    )

    assert result.status == "error"

    assert (
        result.to_message_content()
        == "工具执行失败：模型加载失败"
    )