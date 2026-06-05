"""서빙 백엔드 추상화 — macOS/MLX 또는 Linux/Ollama 를 런타임에 선택한다.

선택 우선순위: 환경변수 GEMMA_BACKEND=mlx|ollama > OS 자동 감지(Darwin→MLX, 그 외→Ollama).

각 backend 는 서빙 제어(start/stop/status)와 자신의 가속기 메트릭(process_metrics)을 책임진다.
전체 시스템 메트릭은 metrics.py 가 담당한다.
"""
import os
import platform
import re
import subprocess
import threading
from pathlib import Path

import httpx

import metrics

BASE_DIR = Path(__file__).parent


def _idle_process() -> dict:
    return {"running": False, "pid": None, "cpu": 0, "mem_gb": 0.0,
            "mem_pct": 0, "mem_total_gb": metrics.TOTAL_GB, "etime": ""}


# ===================== MLX (macOS) =====================
class MLXBackend:
    name = "mlx"
    default_variant = "4bit"
    models = {
        "4bit": "mlx-community/gemma-4-12B-it-4bit",
        "8bit": "mlx-community/gemma-4-12B-it-8bit",
    }

    def chat_base_url(self) -> str:
        return "http://127.0.0.1:8080/v1"

    def _pid(self) -> int | None:
        r = subprocess.run(["pgrep", "-f", "mlx_vlm.server"], capture_output=True, text=True)
        pids = r.stdout.split()
        return int(pids[0]) if pids else None

    def _current_variant(self) -> str | None:
        """실행 중 서버가 로드한 variant (--model 인자에서 파싱). 메인+워커 중 --model 보유 프로세스를 찾는다."""
        r = subprocess.run(["pgrep", "-f", "mlx_vlm.server"], capture_output=True, text=True)
        for pid in r.stdout.split():
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
            for variant, model in self.models.items():
                if model in cmd:
                    return variant
        return None

    def _model_ready(self, variant: str) -> bool:
        """모델 가중치가 HF 캐시에 완전히 받아졌는지 (다운로드 중이면 기동이 블록되므로 가드)."""
        repo = self.models[variant].replace("/", "--")
        hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}"
        snapshots, blobs = hub / "snapshots", hub / "blobs"
        if not snapshots.exists():
            return False
        has_weights = any(snapshots.glob("*/*.safetensors"))
        downloading = blobs.exists() and any(blobs.glob("*.incomplete"))
        return has_weights and not downloading

    def _ready(self) -> bool:
        try:
            return httpx.get(f"{self.chat_base_url()}/models", timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False

    def status(self) -> dict:
        variant = self._current_variant()
        if variant is None:
            return {"state": "stopped", "running": False, "variant": None}
        state = "ready" if self._ready() else "loading"
        return {"state": state, "running": True, "variant": variant}

    def start(self, variant: str) -> dict:
        if variant not in self.models:
            return {"ok": False, "code": 400, "message": f"지원하지 않는 variant: {variant}"}
        if not self._model_ready(variant):
            return {"ok": False, "code": 409, "message": f"{variant} 모델이 아직 다운로드 중입니다. 완료 후 시작하세요."}
        cur = self._current_variant()
        if cur == variant:
            return {"ok": True, "message": "이미 실행 중입니다."}
        if cur is not None:
            subprocess.run(["./stop.sh"], cwd=str(BASE_DIR), capture_output=True, text=True)
        subprocess.Popen(["./start.sh", variant], cwd=str(BASE_DIR),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        switching = "모델 전환 중 — " if cur is not None else ""
        return {"ok": True, "message": f"{switching}{variant} 서버를 시작합니다. 준비까지 수십 초 걸립니다."}

    def stop(self) -> dict:
        if not self._pid():
            return {"ok": True, "message": "실행 중인 서버가 없습니다."}
        subprocess.run(["./stop.sh"], cwd=str(BASE_DIR), capture_output=True, text=True)
        return {"ok": True, "message": "서버를 중지했습니다."}

    def process_metrics(self) -> dict:
        """gemma 프로세스 CPU(raw %) + MEM(footprint phys_footprint) + 실행시간."""
        pid = self._pid()
        if pid is None:
            return _idle_process()
        ps = subprocess.run(["ps", "-o", "%cpu=,etime=", "-p", str(pid)], capture_output=True, text=True).stdout.split()
        cpu = round(float(ps[0])) if ps else 0
        etime = ps[1] if len(ps) > 1 else ""
        mem_gb = 0.0
        fp = subprocess.run(["/usr/bin/footprint", "-p", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s+([\d.]+)\s+(\w+)", fp)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            mem_gb = {"GB": val, "MB": val / 1024, "KB": val / 1048576}.get(unit, val)
        mem_gb = round(mem_gb, 1)
        return {
            "running": True, "pid": pid, "cpu": cpu, "mem_gb": mem_gb,
            "mem_total_gb": metrics.TOTAL_GB,
            "mem_pct": round(mem_gb / metrics.TOTAL_GB * 100) if metrics.TOTAL_GB else 0,
            "etime": etime,
        }


# ===================== Ollama (Linux/CUDA) =====================
class OllamaBackend:
    name = "ollama"
    default_variant = "4bit"
    models = {
        "4bit": "gemma4:12b",
        "8bit": "gemma4:12b-q8_0",
    }
    _api = "http://127.0.0.1:11434"

    def __init__(self) -> None:
        self._starting: str | None = None  # preload 진행 중인 variant (loading 표시용)

    def chat_base_url(self) -> str:
        return f"{self._api}/v1"

    def _reachable(self) -> bool:
        try:
            return httpx.get(f"{self._api}/api/tags", timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False

    def _installed(self, variant: str) -> bool:
        try:
            tags = httpx.get(f"{self._api}/api/tags", timeout=3).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return False
        return any(m.get("name") == self.models[variant] for m in tags)

    def _loaded_variant(self) -> str | None:
        """현재 VRAM 에 로드된 gemma4 모델의 variant (/api/ps)."""
        try:
            running = httpx.get(f"{self._api}/api/ps", timeout=2).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return None
        for m in running:
            for variant, model in self.models.items():
                if m.get("name") == model:
                    return variant
        return None

    def status(self) -> dict:
        if not self._reachable():
            return {"state": "stopped", "running": False, "variant": None}
        variant = self._loaded_variant()
        if variant:
            self._starting = None
            return {"state": "ready", "running": True, "variant": variant}
        if self._starting:
            return {"state": "loading", "running": True, "variant": self._starting}
        return {"state": "stopped", "running": False, "variant": None}

    def _preload(self, variant: str) -> None:
        """모델을 VRAM 에 적재(빈 generate 로 트리거, 30분 keep_alive)."""
        try:
            httpx.post(f"{self._api}/api/generate",
                       json={"model": self.models[variant], "prompt": "", "keep_alive": "30m"},
                       timeout=300)
        except httpx.HTTPError:
            pass

    def start(self, variant: str) -> dict:
        if variant not in self.models:
            return {"ok": False, "code": 400, "message": f"지원하지 않는 variant: {variant}"}
        if not self._reachable():
            return {"ok": False, "code": 503, "message": "ollama 데몬에 연결할 수 없습니다 (systemctl status ollama 확인)."}
        if not self._installed(variant):
            return {"ok": False, "code": 409,
                    "message": f"{self.models[variant]} 모델이 없습니다. `ollama pull {self.models[variant]}` 후 시작하세요."}
        cur = self._loaded_variant()
        if cur == variant:
            return {"ok": True, "message": "이미 로드되어 있습니다."}
        if cur is not None:
            self._unload(cur)  # VRAM 한정 → 다른 variant 는 내린다
        self._starting = variant
        threading.Thread(target=self._preload, args=(variant,), daemon=True).start()
        warn = " ⚠️ 8bit(~13GB)는 12GB VRAM을 초과해 느려질 수 있습니다." if variant == "8bit" else ""
        switching = "모델 전환 중 — " if cur is not None else ""
        return {"ok": True, "message": f"{switching}{variant} 모델을 로드합니다. 준비까지 수십 초 걸립니다.{warn}"}

    def _unload(self, variant: str) -> None:
        try:
            httpx.post(f"{self._api}/api/generate",
                       json={"model": self.models[variant], "keep_alive": 0}, timeout=10)
        except httpx.HTTPError:
            pass

    def stop(self) -> dict:
        self._starting = None
        cur = self._loaded_variant()
        if cur is None:
            return {"ok": True, "message": "로드된 모델이 없습니다."}
        self._unload(cur)
        return {"ok": True, "message": "모델을 언로드했습니다 (VRAM 회수)."}

    def process_metrics(self) -> dict:
        """GPU util(%) + VRAM used/total — '프로세스' 패널에 매핑. nvidia-smi 기반."""
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
        if not out:
            return _idle_process()
        util, used_mib, total_mib = (x.strip() for x in out[0].split(","))
        used_gb = round(int(used_mib) / 1024, 1)
        total_gb = round(int(total_mib) / 1024, 1)
        pid_r = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True).stdout.split()
        return {
            "running": True,
            "pid": int(pid_r[0]) if pid_r else None,
            "cpu": int(util),
            "mem_gb": used_gb,
            "mem_total_gb": total_gb,
            "mem_pct": round(int(used_mib) / int(total_mib) * 100) if int(total_mib) else 0,
            "etime": "GPU",
        }


def pick_backend() -> "MLXBackend | OllamaBackend":
    choice = os.environ.get("GEMMA_BACKEND", "").lower()
    if choice == "mlx":
        return MLXBackend()
    if choice == "ollama":
        return OllamaBackend()
    return MLXBackend() if platform.system() == "Darwin" else OllamaBackend()
