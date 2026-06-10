"""
命令行入口。

这个文件尽量保持简单：
1. 读取用户在终端输入的内容。
2. 调用 run_agent 得到回复。
3. 保存最近几轮对话和关键记忆。

真正的 Agent 逻辑在 agent.py，
真正的文档工具在 document_tools.py。
"""

from agent import run_agent
from config import MAX_HISTORY_PAIRS
from memory import load_memory_state, save_memory_state


# =========================
# 5. 命令行交互入口
# =========================

if __name__ == "__main__":
    print("Agent Demo 已启动。")
    print("你可以输入：现在几点了 / 帮我算 25*17 / 你好")
    print("输入 exit、quit 或 退出 可以结束程序。")
    print("-" * 40)

    # chat_history 保存最近 N 轮自然语言对话。
    # 它只保存 user/assistant 的最终内容，不保存中间 tool_call，避免上下文越来越乱。
    chat_history = []

    # memory_state 保存结构化记忆。
    # 例如用户先问 tech35 的 DRC 章节，下一轮说“继续看 Extract”，
    # Agent 就可以从 current_file 里知道“这个文件”指的是 tech35格式.txt。
    memory_state = load_memory_state()

    while True:
        # input 返回的是字符串，也就是用户这一轮输入的问题。
        user_input = input("你：")

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("助手：已退出。")
            break

        # 把当前输入、最近历史和结构化记忆交给 Agent。
        reply = run_agent(user_input, chat_history, memory_state)

        # Agent 返回最终回答后，再把这一轮 user/assistant 对话写入历史。
        chat_history.append({
            "role": "user",
            "content": user_input
        })
        chat_history.append({
            "role": "assistant",
            "content": reply
        })

        # 只保留最近 MAX_HISTORY_PAIRS 组对话。
        # 一组对话包含 user 和 assistant 两条消息，所以乘以 2。
        chat_history[:] = chat_history[-MAX_HISTORY_PAIRS * 2:]

        save_memory_state(memory_state)

        print("助手：", reply)

        print("-" * 40)
