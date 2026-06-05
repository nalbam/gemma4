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
MODELS = {
    "4bit": "mlx-community/gemma-4-12B-it-4bit",
    "8bit": "mlx-community/gemma-4-12B-it-8bit",
}
DEFAULT_VARIANT = "4bit"
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


def current_variant() -> str | None:
    """실행 중인 서버가 로드한 variant (--model 인자에서 파싱, 없으면 None).
    mlx_vlm.server는 메인+워커 여러 프로세스를 띄우므로, --model 인자를 가진
    프로세스를 찾을 때까지 모든 매칭 프로세스를 확인한다."""
    r = subprocess.run(["pgrep", "-f", "mlx_vlm.server"], capture_output=True, text=True)
    for pid in r.stdout.split():
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        for variant, model in MODELS.items():
            if model in cmd:
                return variant
    return None


def model_ready(variant: str) -> bool:
    """모델 가중치가 캐시에 완전히 받아졌는지 확인한다.
    미완(다운로드 중)이면 mlx_vlm.server가 기동 중 HF 다운로드로 블록되어
    응답·중지가 막히므로, 시작 전에 차단하기 위한 가드."""
    repo = MODELS[variant].replace("/", "--")
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}"
    snapshots, blobs = hub / "snapshots", hub / "blobs"
    if not snapshots.exists():
        return False
    has_weights = any(snapshots.glob("*/*.safetensors"))
    downloading = blobs.exists() and any(blobs.glob("*.incomplete"))
    return has_weights and not downloading


def server_ready() -> bool:
    """gemma 서버가 실제 요청을 받을 수 있는 상태인지 (/v1/models 200).
    프로세스는 떴지만 모델 로딩/다운로드 중이면 응답하지 못해 False."""
    try:
        return httpx.get(f"{GEMMA_URL}/models", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False


def server_state(variant: str | None) -> str:
    """stopped(프로세스 없음) | loading(떴지만 로딩 중) | ready(서빙 가능)."""
    if variant is None:
        return "stopped"
    return "ready" if server_ready() else "loading"


def _used_mem_gb() -> float:
    """실제 사용 메모리 = anonymous + wired + compressed.
    top의 PhysMem 'used'는 inactive·파일 캐시(회수 가능)까지 포함해 과대 표시되므로 쓰지 않는다."""
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    psize = 4096
    m = re.search(r"page size of (\d+)", vm)
    if m:
        psize = int(m.group(1))

    def pages(name: str) -> int:
        mm = re.search(rf"{re.escape(name)}:\s+(\d+)", vm)
        return int(mm.group(1)) if mm else 0

    used_pages = pages("Pages wired down") + pages("Anonymous pages") + pages("Pages occupied by compressor")
    return round(used_pages * psize / 1073741824, 1)


def _system_metrics() -> dict:
    """시스템 CPU(top user+sys) + 실제 사용 메모리(vm_stat)."""
    out = subprocess.run(["top", "-l1", "-n0"], capture_output=True, text=True).stdout
    cpu = 0
    for line in out.splitlines():
        if "CPU usage" in line:
            m = re.search(r"([\d.]+)%\s+idle", line)
            if m:
                cpu = round(100 - float(m.group(1)))
            break
    mem_used_gb = _used_mem_gb()
    return {
        "cpu": cpu,
        "mem_used_gb": mem_used_gb,
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
    return templates.TemplateResponse(
        request, "index.html",
        {"variants": list(MODELS), "default_variant": DEFAULT_VARIANT, "models": MODELS},
    )


@app.get("/api/server/status")
def server_status():
    # running은 모델을 로드한(또는 로드 중인) 메인 프로세스 기준.
    # 중지 직후 잠깐 남는 워커(--model 없음)는 variant=None이라 running=False로 본다.
    variant = current_variant()
    return {"state": server_state(variant), "running": variant is not None, "pid": server_pid(), "variant": variant}


@app.post("/api/server/start")
async def server_start(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    variant = body.get("variant", DEFAULT_VARIANT)
    if variant not in MODELS:
        return JSONResponse(status_code=400, content={"ok": False, "message": f"지원하지 않는 variant: {variant}"})
    if not model_ready(variant):
        return JSONResponse(status_code=409, content={"ok": False, "message": f"{variant} 모델이 아직 다운로드 중입니다. 완료 후 시작하세요."})

    cur = current_variant()
    if cur == variant:
        return {"ok": True, "message": "이미 실행 중입니다."}
    if cur is not None:
        # 다른 variant 실행 중 → 중지 후 재기동 (서버는 단일 모델만 로드)
        subprocess.run(["./stop.sh"], cwd=str(BASE_DIR), capture_output=True, text=True)

    # start.sh는 준비까지 최대 120초 폴링하므로 기다리지 않고 비블로킹 실행
    subprocess.Popen(
        ["./start.sh", variant], cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    switching = "모델 전환 중 — " if cur is not None else ""
    return {"ok": True, "message": f"{switching}{variant} 서버를 시작합니다. 준비까지 수십 초 걸립니다."}


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
    variant = current_variant()
    return {"system": _system_metrics(), "process": process, "ncpu": NCPU, "variant": variant, "state": server_state(variant)}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    variant = current_variant()
    if variant is None:
        return JSONResponse(status_code=502, content={"error": "gemma 서버가 꺼져 있습니다. 먼저 서버를 시작하세요."})
    model = MODELS[variant]

    async def stream():
        # NDJSON: 한 줄에 {"delta": "<토큰 텍스트>", "n": <누적 토큰 수>}
        # mlx 스트리밍은 토큰당 delta를 보내므로 delta 개수 = 생성 토큰 수
        payload = {"model": model, "messages": messages, "max_tokens": 4096, "stream": True}
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
