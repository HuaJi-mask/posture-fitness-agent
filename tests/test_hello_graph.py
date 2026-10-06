from typing import TypedDict
from langgraph.graph import StateGraph, END, START


# 定义状态
class HelloState(TypedDict):
    user_input: str  # 用户输入
    processed_text: str  # 处理后的文本
    final_ouput: str  # 最终输出

def node_uppercase(state: HelloState) -> HelloState:
    """将用户输入转换为大写"""
    print("将用户输入转换为大写")
    input_text = state["user_input"]
    result = input_text.upper()
    # 返回更新后的状态（只需要返回你改了的字段）
    return {"processed_text": result}

def node_respond(state: HelloState) -> HelloState:
    """生成最终回复"""
    print("生成最终回复")
    processed_text = state["processed_text"]
    result = f"你好，{processed_text}"
    return {"final_ouput": result}

def build_hello_graph():
    # 创建状态图
    graph = StateGraph(HelloState)

    # 添加节点
    graph.add_node("_uppercase", node_uppercase)
    graph.add_node("respond", node_respond)

    # 添加边
    graph.add_edge(START, "_uppercase")
    graph.add_edge("_uppercase", "respond")
    # 添加结束节点
    graph.add_edge("respond", END)

    # 编译图：把定义好的图变成可运行的对象
    return graph.compile()

# 测试
if __name__ == "__main__":
    print("="*50)
    print("Langgraph Hello World 测试")
    print("="*50)

    app = build_hello_graph()

    initial_state = {"user_input": "你好"}

    print("开始运行")

    result = app.invoke(initial_state)

    print("运行完成")
    print("="*50)
    print(f"最终输出: {result['final_ouput']}")
    print(f"完整状态: ")
    for key,value in result.items():
        print(f"{key}: {value}")
