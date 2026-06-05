#!/usr/bin/env bash
# gemma4 서버 중지
set -euo pipefail

if pgrep -f mlx_vlm.server >/dev/null; then
  pkill -f mlx_vlm.server
  # 모델 로딩/다운로드 중이면 SIGTERM(graceful)에 둔감 → 최대 5초 대기 후 강제 종료
  for _ in $(seq 1 10); do
    pgrep -f mlx_vlm.server >/dev/null || break
    sleep 0.5
  done
  pgrep -f mlx_vlm.server >/dev/null && pkill -9 -f mlx_vlm.server
  echo "중지했습니다."
else
  echo "실행 중인 서버가 없습니다."
fi
