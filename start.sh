#!/usr/bin/env bash
# gemma4 OpenAI 호환 서버 백그라운드 기동
set -euo pipefail

cd "$(dirname "$0")"

MODEL="mlx-community/gemma-4-12B-it-4bit"
HOST="127.0.0.1"
PORT="8080"
LOG="/tmp/gemma4_server.log"

if pgrep -f mlx_vlm.server >/dev/null; then
  echo "이미 실행 중입니다 (pid: $(pgrep -f mlx_vlm.server | tr '\n' ' '))"
  echo "엔드포인트: http://${HOST}:${PORT}/v1"
  exit 0
fi

echo "서버 기동 중... (모델 로드, 로그: ${LOG})"
nohup python -m mlx_vlm.server \
  --model "${MODEL}" --host "${HOST}" --port "${PORT}" > "${LOG}" 2>&1 &

# 준비될 때까지 대기 (최대 120초)
for _ in $(seq 1 60); do
  if curl -s -o /dev/null "http://${HOST}:${PORT}/v1/models"; then
    echo "준비 완료 ✅  http://${HOST}:${PORT}/v1"
    exit 0
  fi
  sleep 2
done

echo "기동 확인 실패 — 로그를 확인하세요: ${LOG}" >&2
exit 1
