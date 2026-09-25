"""LocalDoc-Agent 命令行入口。"""

from agent import run_agent
from config import MAX_HISTORY_PAIRS


if __name__ == "__main__":
    print("LocalDoc-Agent 已启动。")
    print("把知识库文档放入 data/，然后直接用自然语言提问。")
    print("输入 exit、quit 或 退出 可以结束程序。")
    print("-" * 50)

    # 只保留最近几轮自然语言对话，避免上下文无限增长。
    chat_history = []

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("助手：已退出。")
            break

        if not user_input:
            continue

        reply = run_agent(user_input, chat_history)

        chat_history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        chat_history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        chat_history[:] = chat_history[-MAX_HISTORY_PAIRS * 2:]

        print("助手：", reply)
        print("-" * 50)
