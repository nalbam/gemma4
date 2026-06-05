#!/usr/bin/env bash
# WSL/CUDA 부트스트랩 — Ollama 버전 확인 + gemma4 모델 다운로드 + Python 의존성.
set -euo pipefail

cd "$(dirname "$0")/.."   # 프로젝트 루트 (requirements-wsl.txt 위치)

MODEL="gemma4:12b"
MIN_VER="0.20"

echo "== Ollama 버전 확인 =="
ver="$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo 0)"
echo "현재: ${ver:-미설치}"
need=$(awk -v v="$ver" -v m="$MIN_VER" 'BEGIN{
  split(v, a, "."); split(m, b, ".");
  print (a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])) ? 1 : 0
}')
if [ "$need" = "1" ]; then
  echo "⚠️  Gemma 4 는 Ollama ${MIN_VER}+ 가 필요합니다. 업데이트 후 다시 실행하세요 (sudo 필요):"
  echo "    curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

echo "== 모델 다운로드: ${MODEL} (~7GB) =="
ollama pull "${MODEL}"

echo "== Python 의존성 설치 =="
pip install -r requirements-wsl.txt

echo "완료 ✅"
echo "  웹 UI :  python webapp.py        # http://127.0.0.1:8000"
echo "  채팅  :  python chat.py \"안녕\"     # 또는 대화형: python chat.py"
