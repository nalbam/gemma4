"""MLX 백엔드 (macOS) — mlx_vlm.server(:8080) 제어 + footprint 메트릭."""
import re
import subprocess
from pathlib import Path

import httpx

import metrics
from .base import idle_process

SCRIPTS = Path(__file__).parent.parent / "scripts"


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
            subprocess.run(["./stop.sh"], cwd=str(SCRIPTS), capture_output=True, text=True)
        subprocess.Popen(["./start.sh", variant], cwd=str(SCRIPTS),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        switching = "모델 전환 중 — " if cur is not None else ""
        return {"ok": True, "message": f"{switching}{variant} 서버를 시작합니다. 준비까지 수십 초 걸립니다."}

    def stop(self) -> dict:
        if not self._pid():
            return {"ok": True, "message": "실행 중인 서버가 없습니다."}
        subprocess.run(["./stop.sh"], cwd=str(SCRIPTS), capture_output=True, text=True)
        return {"ok": True, "message": "서버를 중지했습니다."}

    def process_metrics(self) -> dict:
        """gemma 프로세스 CPU(raw %) + MEM(footprint phys_footprint) + 실행시간."""
        pid = self._pid()
        if pid is None:
            return idle_process()
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
