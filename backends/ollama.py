"""Ollama 백엔드 (Linux/CUDA) — :11434 제어 + nvidia-smi GPU 메트릭."""
import subprocess
import threading

import httpx

import metrics
from .base import idle_process


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
            return idle_process()
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
