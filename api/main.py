"""
FastAPI 接口
"""
import asyncio
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import List, Dict
from sqlalchemy.orm import Session
from src.database.connection import init_db, get_db
from src.database import crud
from src.utils.llm import JsonResponseStreamer

from src.graph.main_graph import build_main_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库"""
    init_db()
    print("数据库初始化完成")
    yield


# 创建FastAPI应用
app = FastAPI(title="体态矫正和健身计划智能体", version="1.0.0", lifespan=lifespan)

# CORS：允许前端（file:// 或任意端口）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管前端静态文件
app.mount("/static", StaticFiles(directory="web"), name="static")

# 构建图
graph = build_main_graph()

# ==================== 启动事件 ====================
# 数据库初始化已改为 lifespan（@app.on_event 在 FastAPI 新版中已弃用）


# ==================== 请求/响应模型 ====================
class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    response: str
    intent: str
    stage: str


class Message(BaseModel):
    """消息模型"""
    role: str
    content: str


class HistoryResponse(BaseModel):
    """历史记录响应"""
    session_id: str
    messages: List[Message]
    user_profile: Dict

# ==================== 工具函数 ====================

# def _get_or_create_session(session_id: str) -> dict:
#     """获取或创建会话状态"""
#     if session_id not in sessions:
#         sessions[session_id] = {
#             "messages": [],
#             "conversation_round": 0,
#             "user_profile": {}
#         }
#     return sessions[session_id]

def load_state_from_db(db: Session, session_id: str) -> dict:
    """从数据库加载会话状态"""
    # 获取或创建会话
    crud.get_or_create_session(db, session_id)
    # 获取历史消息
    messages = crud.get_messages(db, session_id)
    # 获取用户信息
    user_profile = crud.get_session_profile(db, session_id)
    # 获取最新方案
    latest_plan = crud.get_latest_plan(db, session_id)
    state = {
        "messages": messages,
        "conversation_round": len(messages),
        "user_profile": user_profile,
        "training_plan": latest_plan.get("training_plan", ""),
        "diet_plan": latest_plan.get("diet_plan", ""),
        "validation_report": latest_plan.get("validation_report", ""),
    }

    return state

def save_state_to_db(db: Session, session_id: str, state: dict):
    """保存会话状态到数据库"""
    # 更新用户信息
    user_profile = state.get("user_profile", {})
    crud.update_session_profile(db, session_id, user_profile)

    # 保存方案（如果有）
    training_plan = state.get("training_plan", "")
    diet_plan = state.get("diet_plan", "")
    validation_report = state.get("validation_report", "")
    if training_plan:
        latest_plan = crud.get_latest_plan(db, session_id)
        if not latest_plan or latest_plan["training_plan"] != training_plan:
            crud.save_plan(db, session_id, training_plan, diet_plan, validation_report)

# ==================== 接口 ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    发送消息，获取回复
    """
    session_id = request.session_id
    user_message = request.message


    if not user_message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    state = load_state_from_db(db, session_id)

    # 把用户消息加到state里
    state["user_input"] = user_message
    state["messages"].append({
        "role": "user",
        "content": user_message
    })

    # 运行图（agent 节点已改为异步，必须用 ainvoke）
    try:
        result = await graph.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")

    # 存回数据库
    # 存用户消息
    crud.add_message(db, session_id, "user", user_message)
    # 存助手消息
    ai_response = result.get("current_response", "")
    crud.add_message(db, session_id, "assistant", ai_response)

    # 更新会话标题和最后消息
    crud.update_session_info(db, session_id, title=user_message, last_message=ai_response)

    # 存用户画像和方案
    save_state_to_db(db, session_id, result)

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        intent=result.get("intent", "unknown"),
        stage=result.get("current_stage", "unknown")
    )

    



@app.get("/history", response_model=HistoryResponse)
def get_history(session_id: str, db: Session = Depends(get_db)):
    """
    获取会话历史记录

    注意：新建会话时后端还没有会话记录（首次发消息才 get_or_create_session），
    此时查历史返回空列表而不是 404——"新会话无历史"是合法状态，不是错误
    """
    # 从数据库加载消息（get_messages 按 session_id 直接查消息表，不依赖会话记录存在）
    messages = crud.get_messages(db, session_id)
    user_profile = crud.get_session_profile(db, session_id)

    return HistoryResponse(
        session_id=session_id,
        messages=[Message(role=msg["role"], content=msg["content"]) for msg in messages],
        user_profile=user_profile
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    """
    流式聊天接口（SSE 格式）
    """
    session_id = request.session_id
    user_message = request.message

    if not user_message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    state = load_state_from_db(db, session_id)

    # 把用户消息加到state里
    state["user_input"] = user_message
    state["messages"].append({
        "role": "user",
        "content": user_message
    })

    # 生成器函数，流式返回
    async def event_generator():
        final_state = None
        ai_response = ""

        try:
            # 双流式模式：
            # - "messages" 模式：agent 异步调用 LLM 时逐 token 吐出的原始 chunk
            #   （metadata.langgraph_node 标识属于哪个节点）。agent 输出的是 JSON，
            #   不能直接给用户看，用 JsonResponseStreamer 增量提取 response 字段
            #   的可见文本，转成 token 事件
            # - "debug" 模式：task 事件（节点开始）→ status 事件；
            #   task_result 事件（节点结束）→ node 事件（带权威的完整回复）
            streamers = {}  # node_name -> JsonResponseStreamer
            node_runs = {}  # node_name -> 当前运行次数（并行/辩论循环会重复运行同一节点）
            node_streamed = {}  # node_name -> 本轮是否有 token 流式输出（node 事件带权威标记）

            async for mode, payload in graph.astream(state, stream_mode=["messages", "debug"]):
                if mode == "messages":
                    chunk, meta = payload
                    node_name = meta.get("langgraph_node", "")
                    content = chunk.content
                    # triage/merge_validation 等无可见文本的节点，提取器自然输出空串
                    if node_name and isinstance(content, str):
                        streamer = streamers.setdefault(node_name, JsonResponseStreamer())
                        # token 事件带上节点名和运行序号——并行时多个节点（如体态/健身
                        # 专家）的 token 会交错到达，前端需要靠 (node, run) 区分气泡，
                        # 否则不同节点的文本会混进同一个气泡
                        node_run = node_runs.get(node_name, 1)
                        streamed_any = False
                        for text in streamer.feed(content):
                            streamed_any = True
                            yield f"data: {json.dumps({'type': 'token', 'node': node_name, 'run': node_run, 'token': text}, ensure_ascii=False)}\n\n"
                        if streamed_any:
                            node_streamed.setdefault(node_name, set()).add(node_run)
                elif mode == "debug":
                    etype = payload.get("type")
                    inner = payload.get("payload", {})
                    node_name = inner.get("name", "")

                    if etype == "task":
                        # 节点开始：立即更新前端进度（节点可能要跑几十秒，不用等它跑完）
                        if node_name:
                            # 该节点本轮运行序号+1（前端用它区分并行/辩论循环中的同一节点）
                            node_runs[node_name] = node_runs.get(node_name, 0) + 1
                            # 重置该节点的提取器——辩论循环会重复运行同一节点，
                            # 若复用旧提取器，emitted 还停在上一轮进度，新一轮回复
                            # 只有超出旧长度的部分才会流出，用户只能看到残片
                            streamers.pop(node_name, None)
                            yield f"data: {json.dumps({'type': 'status', 'node': node_name}, ensure_ascii=False)}\n\n"
                    elif etype == "task_result":
                        # 节点结束：发该节点的权威回复（流式提取异常时前端用它兜底）
                        node_output = inner.get("result") or {}
                        node_run = node_runs.get(node_name, 1)
                        # 该节点本轮是否流过 token（前端据此决定是否用完整回复兜底；
                        # 并行时前端自己的"本节点是否流过"标志会被其他节点污染，
                        # 必须以这里的权威标记为准）
                        streamed = node_run in node_streamed.get(node_name, set())
                        sse_event = {
                            'type': 'node',
                            'node': node_name,
                            'run': node_run,
                            'streamed': streamed,
                            'data': {
                                'stage': node_output.get('current_stage', ''),
                                'intent': node_output.get('intent', ''),
                                'response': node_output.get('current_response', '')
                            }
                        }
                        # merge_expert_responses 的回复 = 两个专家回复的拼接，专家内容
                        # 已各自流式显示过，前端遇到 dedup 且本轮已有流式气泡时跳过，
                        # 避免用户看到重复内容（无流式时仍作为兜底显示）
                        if node_name == "merge_expert_responses":
                            sse_event['dedup'] = True
                        yield f"data: {json.dumps(sse_event, ensure_ascii=False)}\n\n"
                        final_state = node_output
        except asyncio.CancelledError:
            # 用户点击了停止，客户端断开连接。
            # 不保存数据库；必须重新抛出（不能吞掉）：
            # CancelledError 与 graph.astream 迭代器在同一个协程栈上，若在此
            # return 吞掉异常，迭代器不会被取消，进行中的 LLM 调用会继续跑完
            # （白白消耗 token，只是结果不落库）；re-raise 让取消信号传播到
            # LLM 调用，真正中断图执行（实测验证：吞掉时"分析完成"仍会打印）
            print("【流式接口】用户中断了请求")
            raise
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return

        # 流结束：存回数据库
        try:
            final_response = ai_response or (final_state.get("current_response", "") if final_state else "")
            crud.add_message(db, session_id, "user", user_message)
            crud.add_message(db, session_id, "assistant", final_response)
            if final_state:
                save_state_to_db(db, session_id, final_state)
            crud.update_session_info(db, session_id, title=user_message, last_message=final_response)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return

        # 结束标记
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/sessions")
async def list_session(db: Session = Depends(get_db)):
    """获取所有会话列表（按更新时间倒序）"""
    sessions = crud.get_all_sessions(db)
    return {"sessions": sessions}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """删除会话（幂等：删除不存在的会话也返回成功）

    用户可能删除一个从未发过消息的空会话（前端新建了会话但没发消息就删除），
    此时后端尚无会话记录，不该报 404
    """
    crud.delete_session(db, session_id)
    return {"success": True, "message": "会话删除成功"}


@app.get("/")
async def serve_frontend():
    """根路径返回前端页面"""
    return FileResponse("web/index.html")


# ==================== 运行 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
