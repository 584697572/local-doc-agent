"""
工具 Schema 定义。

这里的 TOOLS 会传给大模型，作用是告诉模型：
1. 有哪些工具可以调用。
2. 每个工具适合什么场景。
3. 调用工具时需要传哪些参数。

注意：
这里不会真正执行工具。
真正执行工具的是 agent.py 里的 ToolRegistry。
所以每个 schema 里的 "name" 必须和 ToolRegistry 注册的工具名保持一致。
"""

# =========================
# 2. 写工具说明，告诉模型有哪些工具可用
# =========================

# OpenAI/DeepSeek function calling 要求工具说明使用这种 JSON-like 字典结构。
# 模型会根据 description 和 parameters 自动决定是否生成 tool_calls。
TOOLS = [
    {
        # type=function 表示这是一个函数工具
        "type": "function",

        "function": {
            # 工具名字
            # 这个名字要和我们后面判断时使用的名字一致
            "name": "get_current_time",

            # 工具说明
            # 模型会根据 description 判断什么时候该调用这个工具
            "description": "当用户询问当前时间、现在几点、今天日期、当前日期时间时，调用这个工具。",

            # 参数说明
            # get_current_time 不需要任何参数，所以 properties 为空
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",

        "function": {
            "name": "calculate",

            "description": "当用户要求进行数学计算时，调用这个工具。例如计算 25*17、100+200、3.14*2。",

            # calculate 需要一个参数 expression
            "parameters": {
                "type": "object",

                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式，例如：'25*17'、'100+200'"
                    }
                },

                # required 表示 expression 是必填参数
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",

        "function": {
            "name": "get_today_date",

            "description": "当用户询问今天日期、今天几号、今天是几月几日时，调用这个工具。",

            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
    "type": "function",

    "function": {
        "name": "read_text_file",

        "description": "当用户要求读取、查看、总结、分析某个 txt 文本文档时，调用这个工具。",

        "parameters": {
            "type": "object",

            "properties": {
                "filename": {
                    "type": "string",
                    "description": "要读取的 txt 文件名，例如 demo.txt、notes.txt、paper_summary.txt"
                }
            },

            "required": ["filename"]
        }
    }
    
    },
    {
    "type": "function",

    "function": {
        "name": "list_txt_files",

        "description": "当用户想查看当前目录有哪些 txt 文件，或者用户没有说清楚具体文件名时，调用这个工具。",

        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    },
    {
    "type": "function",

    "function": {
        "name": "search_text_file",

        "description": (
            "当用户想在某个 txt 文件中查找某个关键词、章节、术语、字段或内容位置时，调用这个工具。"
            "它会返回关键词附近的上下文，而不是读取整个文件。"
        ),

        "parameters": {
            "type": "object",

            "properties": {
                "filename": {
                    "type": "string",
                    "description": "要搜索的 txt 文件名或文件名关键词，例如 tech35、tech35格式.txt、读论文"
                },
                "keyword": {
                    "type": "string",
                    "description": "要在文件中搜索的关键词，例如 Drc section、Extract section、Contact section"
                }
            },

            "required": ["filename", "keyword"]
        }
    }
    },
    {
        "type": "function",

        "function": {
            "name": "extract_text_section",

            "description": (
                "当用户想查看、了解、总结某个 txt 文件中的具体章节时，调用这个工具。"
                "例如 DRC 章节、Extract 章节、Contact 章节。"
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "要提取章节的 txt 文件名或文件名关键词，例如 tech35、tech35格式.txt"
                    },
                    "section_name": {
                        "type": "string",
                        "description": "要提取的章节名，例如 Drc、Extract、Contact"
                    }
                },

                "required": ["filename", "section_name"]
            }
        }
    }
    

]
