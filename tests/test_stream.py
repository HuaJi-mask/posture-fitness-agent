"""SSE 流式接口测试：逐条打印事件，观察流式效果"""
import json
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/chat/stream",
    json={"session_id": "demo1", "message": "我脖子疼，办公室坐久了"},
    timeout=120,
) as resp:
    print(f"HTTP {resp.status_code}")
    for line in resp.iter_lines():
        if line.startswith("data:"):
            data = json.loads(line[5:].strip())
            print(f"收到 {data.get('type')}: node={data.get('node')} "  
                f"stage={data.get('data', {}).get('stage')} "
                f"response={data.get('data', {}).get('response', '')[:30]}")