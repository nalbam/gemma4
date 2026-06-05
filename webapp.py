"""gemma4 웹 관리 UI — 서버 제어 + 자원 모니터링 + 채팅을 한 화면에서.

실행: python webapp.py  →  http://127.0.0.1:8000
"""
import json
import re
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).parent
GEMMA_URL = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/gemma-4-12B-it-4bit"
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000

app = FastAPI(title="gemma4 web")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 시스템 총 메모리 / 코어 수 (기동 시 1회)
TOTAL_BYTES = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
TOTAL_GB = round(TOTAL_BYTES / 1073741824, 1)
NCPU = int(subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip())


def server_pid() -> int | None:
    """실행 중인 mlx_vlm.server PID (없으면 None)."""
    r = subprocess.run(["pgrep", "-f", "mlx_vlm.server"], capture_output=True, text=True)
    pids = r.stdout.split()
    return int(pids[0]) if pids else None


def _system_metrics() -> dict:
    """top -l1 한 번으로 시스템 CPU(user+sys) + MEM(used) 수집."""
    out = subprocess.run(["top", "-l1", "-n0"], capture_output=True, text=True).stdout
    cpu = 0
    mem_used_gb = 0.0
    for line in out.splitlines():
        if "CPU usage" in line:
            m = re.search(r"([\d.]+)%\s+idle", line)
            if m:
                cpu = round(100 - float(m.group(1)))
        elif line.startswith("PhysMem"):
            m = re.search(r"PhysMem:\s+([\d.]+)([GM])\s+used", line)
            if m:
                val = float(m.group(1))
                mem_used_gb = val / 1024 if m.group(2) == "M" else val
    return {
        "cpu": cpu,
        "mem_used_gb": round(mem_used_gb, 1),
        "mem_total_gb": TOTAL_GB,
        "mem_pct": round(mem_used_gb / TOTAL_GB * 100) if TOTAL_GB else 0,
    }


def _process_metrics(pid: int) -> dict:
    """gemma 프로세스 CPU(raw %) + MEM(footprint phys_footprint) + 실행시간."""
    ps = subprocess.run(["ps", "-o", "%cpu=,etime=", "-p", str(pid)], capture_output=True, text=True).stdout.split()
    cpu = round(float(ps[0])) if ps else 0
    etime = ps[1] if len(ps) > 1 else ""
    # ps RSS는 MLX/Metal 메모리를 못 잡으므로 footprint 사용
    mem_gb = 0.0
    fp = subprocess.run(["/usr/bin/footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s+([\d.]+)\s+(\w+)", fp)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        mem_gb = {"GB": val, "MB": val / 1024, "KB": val / 1048576}.get(unit, val)
    return {
        "running": True,
        "pid": pid,
        "cpu": cpu,
        "mem_gb": round(mem_gb, 1),
        "mem_pct": round(mem_gb / TOTAL_GB * 100) if TOTAL_GB else 0,
        "etime": etime,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"model": MODEL})


@app.get("/api/server/status")
def server_status():
    pid = server_pid()
    return {"running": pid is not None, "pid": pid}


@app.post("/api/server/start")
def server_start():
    if server_pid():
        return {"ok": True, "message": "이미 실행 중입니다."}
    # start.sh는 준비까지 최대 120초 폴링하므로 기다리지 않고 비블로킹 실행
    subprocess.Popen(
        ["./start.sh"], cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return {"ok": True, "message": "서버를 시작합니다. 준비까지 수십 초 걸립니다."}


@app.post("/api/server/stop")
def server_stop():
    if not server_pid():
        return {"ok": True, "message": "실행 중인 서버가 없습니다."}
    subprocess.run(["./stop.sh"], cwd=str(BASE_DIR), capture_output=True, text=True)
    return {"ok": True, "message": "서버를 중지했습니다."}


@app.get("/api/metrics")
def metrics():
    pid = server_pid()
    process = _process_metrics(pid) if pid else {"running": False, "pid": None, "cpu": 0, "mem_gb": 0.0, "mem_pct": 0, "etime": ""}
    return {"system": _system_metrics(), "process": process, "ncpu": NCPU}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    if server_pid() is None:
        return JSONResponse(status_code=502, content={"error": "gemma 서버가 꺼져 있습니다. 먼저 서버를 시작하세요."})

    async def stream():
        # NDJSON: 한 줄에 {"delta": "<토큰 텍스트>", "n": <누적 토큰 수>}
        # mlx 스트리밍은 토큰당 delta를 보내므로 delta 개수 = 생성 토큰 수
        payload = {"model": MODEL, "messages": messages, "max_tokens": 4096, "stream": True}
        n = 0
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", f"{GEMMA_URL}/chat/completions", json=payload) as resp:
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
