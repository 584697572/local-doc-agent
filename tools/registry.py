"""轻量工具注册与执行。"""


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
            return f"未知工具：{name}"

        arg_names = tool["arg_names"]
        missing_args = [
            arg_name
            for arg_name in arg_names
            if arg_name not in arguments
        ]

        if missing_args:
            return "工具参数缺失：" + ", ".join(missing_args)

        kwargs = {
            arg_name: arguments[arg_name]
            for arg_name in arg_names
        }

        return tool["func"](**kwargs)
