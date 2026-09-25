"""提供给 LLM 的 Tool Calling Schema。"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "在本地知识库中检索与用户问题最相关的证据。"
                "当问题依赖本地文档中的事实、概念或技术内容时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于知识库检索的查询语句，保留关键实体和术语。",
                    }
                },
                "required": ["query"],
            },
        },
    }
]
