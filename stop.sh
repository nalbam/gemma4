#!/usr/bin/env bash
# gemma4 서버 중지
set -euo pipefail

if pgrep -f mlx_vlm.server >/dev/null; then
  pkill -f mlx_vlm.server
  echo "중지했습니다."
else
  echo "실행 중인 서버가 없습니다."
fi
