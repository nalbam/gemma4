#!/usr/bin/env bash
# gemma4 서버 자원(CPU/MEM) 모니터링. 인자: 갱신 간격(초, 기본 2)
set -euo pipefail

INTERVAL="${1:-2}"

while true; do
  pid="$(pgrep -f mlx_vlm.server | head -1 || true)"
  clear
  echo "gemma4 모니터  $(date '+%Y-%m-%d %H:%M:%S')  (${INTERVAL}s 간격, Ctrl-C 종료)"
  echo "------------------------------------------------------------"
  if [ -z "${pid}" ]; then
    echo "서버 미실행 (mlx_vlm.server 프로세스 없음)"
  else
    echo "[서버 프로세스]"
    printf "%-7s %-6s %-6s %-9s %s\n" "PID" "%CPU" "%MEM" "RSS(GB)" "ELAPSED"
    ps -o pid=,%cpu=,%mem=,rss=,etime= -p "${pid}" \
      | awk '{printf "%-7s %-6s %-6s %-9.2f %s\n", $1, $2, $3, $4/1048576, $5}'
  fi
  echo
  echo "[시스템 메모리]"
  top -l 1 -n 0 | grep PhysMem
  sleep "${INTERVAL}"
done
