"""轻量工具注册与执行。"""

from tools.result import ToolResult

class ToolRegistry:
    """维护 tool name 到 Python 函数的映射。"""

    def __init__(self):
        self._tools = {}

    def register(self, name, func, arg_names=None):
        self._tools[name] = {
            "func": func,
            "arg_names": arg_names or [],
        }

    def run(self, name, arguments):
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
        error=f"未知工具：{name}"
    )

        arg_names = tool["arg_names"]
        missing_args = [
            arg_name
            for arg_name in arg_names
            if arg_name not in arguments
        ]

        if missing_args:
            return ToolResult.failure(
                error=(
                    "工具参数缺失："
                    + ", ".join(missing_args)
                )
            )

        kwargs = {
            arg_name: arguments[arg_name]
            for arg_name in arg_names
        }

        try:
            result = tool["func"](**kwargs)

        except Exception as exc:
            return ToolResult.failure(
                error=f"工具执行异常：{exc}"
            )

        # 正式 Tool 应该返回 ToolResult
        if isinstance(result, ToolResult):
            return result

        # 兼容未来可能返回普通字符串的简单 Tool
        return ToolResult.success(
            content=str(result)
        )
