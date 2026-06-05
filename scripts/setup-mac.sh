#!/usr/bin/env bash
# macOS/MLX 부트스트랩 — Python(brew) + MLX 패키지(pip) + gemma4 모델.
set -euo pipefail

cd "$(dirname "$0")/.."   # 프로젝트 루트 (requirements.txt 위치)

MODEL="mlx-community/gemma-4-12B-it-4bit"

echo "== Python 확인 =="
if ! command -v python3 >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Python 설치 (brew)..."
    brew install python@3.12
  else
    echo "⚠️  Homebrew가 없습니다. https://brew.sh 설치 후 다시 실행하세요 (또는 python@3.12 직접 설치)." >&2
    exit 1
  fi
fi
python3 --version

echo "== MLX 패키지 설치 (pip — mlx/mlx-vlm 은 PyPI) =="
pip install -r requirements.txt

echo "== 모델 다운로드: ${MODEL} (~7GB) =="
hf download "${MODEL}"

echo "완료 ✅"
echo "  서버  :  ./scripts/start.sh      # http://127.0.0.1:8080/v1"
echo "  웹 UI :  python webapp.py        # http://127.0.0.1:8000"
echo "  채팅  :  python chat.py \"안녕\""
