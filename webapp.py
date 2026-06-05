"""gemma4 웹 관리 UI — 서버 제어 + 자원 모니터링 + 채팅을 한 화면에서.

서빙 백엔드(macOS/MLX 또는 Linux/Ollama)는 backend.py 가 런타임에 선택한다.
실행: python webapp.py  →  http://127.0.0.1:8000
"""
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

import metrics
from backend import pick_backend

BASE_DIR = Path(__file__).parent
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000

backend = pick_backend()

app = FastAPI(title="gemma4 web")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html",
        {"variants": list(backend.models), "default_variant": backend.default_variant, "models": backend.models},
    )


@app.get("/api/server/status")
def server_status():
    st = backend.status()
    return {"state": st["state"], "running": st["running"], "variant": st["variant"], "backend": backend.name}


@app.post("/api/server/start")
async def server_start(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    r = backend.start(body.get("variant", backend.default_variant))
    if not r.get("ok"):
        return JSONResponse(status_code=r.get("code", 400), content={"ok": False, "message": r["message"]})
    return {"ok": True, "message": r["message"]}


@app.post("/api/server/stop")
def server_stop():
    return backend.stop()


@app.get("/api/metrics")
def metrics_endpoint():
    st = backend.status()
    process = backend.process_metrics() if st["running"] else {
        "running": False, "pid": None, "cpu": 0, "mem_gb": 0.0,
        "mem_pct": 0, "mem_total_gb": metrics.TOTAL_GB, "etime": "",
    }
    return {"system": metrics.system(), "process": process, "ncpu": metrics.NCPU,
            "variant": st["variant"], "state": st["state"]}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    st = backend.status()
    if not st["running"] or st["variant"] is None:
        return JSONResponse(status_code=502, content={"error": "gemma 서버가 꺼져 있습니다. 먼저 서버를 시작하세요."})
    model = backend.models[st["variant"]]
    chat_url = f"{backend.chat_base_url()}/chat/completions"

    async def stream():
        # NDJSON: 한 줄에 {"delta": "<토큰 텍스트>", "n": <누적 토큰 수>}
        payload = {"model": model, "messages": messages, "max_tokens": 4096, "stream": True}
        n = 0
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", chat_url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta:
                            n += 1
                            yield json.dumps({"delta": delta, "n": n}, ensure_ascii=False) + "\n"
        except httpx.HTTPError as e:
            yield json.dumps({"delta": f"\n[오류] gemma 서버 통신 실패: {e}", "n": n}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson; charset=utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
