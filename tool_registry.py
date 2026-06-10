"""
简化版工具注册表。

大模型返回 tool_calls 时，只会告诉我们“我要调用哪个工具”和“参数是什么”。
真正把工具名映射到 Python 函数、并执行函数的，是这个 ToolRegistry。

这样做的好处是：
1. 新增工具时，只需要注册一次，不用在 agent.py 里继续写一长串 if/elif。
2. 工具名、函数、参数名集中管理，更容易检查和维护。
3. 模型多传了无关参数时，这里会自动忽略，只把函数需要的参数传进去。
"""


class ToolRegistry:
    """
    一个很轻量的工具注册表。

    内部用字典保存工具：
    {
        "工具名": {
            "func": 真正执行的 Python 函数,
            "arg_names": 这个函数需要从 arguments 里取哪些参数
        }
    }
    """

    def __init__(self):
        # _tools 是内部字典，不希望外部代码直接修改，所以用下划线开头表示“内部使用”。
        self._tools = {}

    def register(self, name, func, arg_names=None):
        """
        注册一个工具。

        name:
            模型调用时使用的工具名，必须和 tool_schemas.py 里的 name 一致。

        func:
            真正执行工作的 Python 函数。

        arg_names:
            这个函数需要哪些参数。
            例如 calculate 需要 ["expression"]，read_text_file 需要 ["filename"]。
            如果工具不需要参数，比如 get_current_time，就可以不传。
        """

        self._tools[name] = {
            "func": func,
            "arg_names": arg_names or [],
        }

    def run(self, name, arguments):
        """
        根据工具名执行对应函数。

        arguments 是模型传回来的参数字典，例如：
        {"filename": "tech35格式.txt", "section_name": "Drc"}
        """

        tool = self._tools.get(name)
        if tool is None:
            return f"未知工具：{name}"

        func = tool["func"]
        arg_names = tool["arg_names"]

        # 只提取当前函数声明需要的参数。
        # 这样即使模型多传了别的字段，也不会导致 Python 函数因为多余参数报错。
        kwargs = {
            arg_name: arguments.get(arg_name, "")
            for arg_name in arg_names
        }

        return func(**kwargs)
