from agent import run_agent
from config import MAX_HISTORY_PAIRS


# =========================
# 5. 命令行交互入口
# =========================

if __name__ == "__main__":
    print("Agent Demo 已启动。")
    print("你可以输入：现在几点了 / 帮我算 25*17 / 你好")
    print("输入 exit、quit 或 退出 可以结束程序。")
    print("-" * 40)

    chat_history = []
    memory_state = {
        "current_file": None,
        "current_section": None,
    }

    while True:
        user_input = input("你：")

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("助手：已退出。")
            break

        reply = run_agent(user_input, chat_history, memory_state)

        chat_history.append({
            "role": "user",
            "content": user_input
        })
        chat_history.append({
            "role": "assistant",
            "content": reply
        })

        chat_history[:] = chat_history[-MAX_HISTORY_PAIRS * 2:]

        print("助手：", reply)

        print("-" * 40)