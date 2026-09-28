"""ToolRegistry 的基础行为测试。"""

from tools.registry import ToolRegistry

def test_registry_runs_registered_tool():
    registry = ToolRegistry()
    registry.register("echo", lambda text: text, ["text"])

    result = registry.run(
        "echo",
        {"text": "hello"},
    )

    assert result.status == "success"
    assert result.content == "hello"


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry()

    result = registry.run("missing", {})

    assert result.status == "error"
    assert "未知工具" in result.error


def test_registry_reports_missing_argument():
    registry = ToolRegistry()
    registry.register("echo", lambda text: text, ["text"])

    result = registry.run("echo", {})

    assert result.status == "error"
    assert "工具参数缺失" in result.error
