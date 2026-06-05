"""서빙 백엔드 추상화 — macOS/MLX 또는 Linux/Ollama 를 런타임에 선택한다.

선택 우선순위: 환경변수 GEMMA_BACKEND=mlx|ollama > OS 자동 감지(Darwin→MLX, 그 외→Ollama).

각 backend 는 서빙 제어(start/stop/status)와 자신의 가속기 메트릭(process_metrics)을 책임진다.
전체 시스템 메트릭은 metrics.py 가 담당한다.
"""
import os
import platform

from .base import idle_process
from .mlx import MLXBackend
from .ollama import OllamaBackend

__all__ = ["MLXBackend", "OllamaBackend", "idle_process", "pick_backend"]


def pick_backend() -> "MLXBackend | OllamaBackend":
    choice = os.environ.get("GEMMA_BACKEND", "").lower()
    if choice == "mlx":
        return MLXBackend()
    if choice == "ollama":
        return OllamaBackend()
    return MLXBackend() if platform.system() == "Darwin" else OllamaBackend()
