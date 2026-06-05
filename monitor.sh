#!/usr/bin/env bash
# gemma4 서버 자원 모니터링 — 전체 시스템 / gemma4 프로세스의 CPU·MEM 막대 그래프.
# 인자: 갱신 간격(초, 기본 2)
set -uo pipefail

INTERVAL="${1:-2}"
WIDTH=30

GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

# bar <percent-int> <suffix>
bar() {
  local pct=$1 suffix="${2:-}"
  (( pct < 0 )) && pct=0
  local cap=$pct; (( cap > 100 )) && cap=100
  local filled=$(( cap * WIDTH / 100 )) empty
  empty=$(( WIDTH - filled ))
  local color=$GREEN
  if   (( pct >= 85 )); then color=$RED
  elif (( pct >= 60 )); then color=$YELLOW
  fi
  local i
  printf '%s' "$color"
  for ((i = 0; i < filled; i++)); do printf '█'; done
  printf '%s%s' "$RESET" "$DIM"
  for ((i = 0; i < empty; i++)); do printf '░'; done
  printf '%s %3d%%  %s' "$RESET" "$pct" "$suffix"
}

total_bytes=$(sysctl -n hw.memsize)
total_gb=$(awk -v b="$total_bytes" 'BEGIN{printf "%.1f", b/1073741824}')
ncpu=$(sysctl -n hw.ncpu)

trap 'printf "\033[?25h\n"' EXIT      # 종료 시(어떤 경로든) 커서 복원
trap 'exit 130' INT TERM             # Ctrl+C/TERM → 실제 종료 (EXIT trap이 커서 복원)
printf '\033[2J\033[?25l'            # 전체 클리어 1회 + 커서 숨김

while true; do
  pid="$(pgrep -f mlx_vlm.server | head -1 || true)"

  # 전체 시스템 (top 1회 호출로 CPU + MEM 동시 수집)
  topout="$(top -l1 -n0)"
  sys_cpu=$(awk '/CPU usage/{v=$7; gsub(/%/,"",v); printf "%d", 100-v+0.5}' <<< "$topout")
  # 실제 사용 = anon + wired + compressed (top의 PhysMem used는 회수 가능한 캐시까지 포함해 과대)
  used_gb=$(vm_stat | awk '
    /page size of/ { match($0, /[0-9]+/); ps = substr($0, RSTART, RLENGTH) }
    /Pages wired down/ { gsub(/\./, ""); w = $NF }
    /Anonymous pages/ { gsub(/\./, ""); a = $NF }
    /Pages occupied by compressor/ { gsub(/\./, ""); c = $NF }
    END { printf "%.1f", (w + a + c) * ps / 1073741824 }')
  sys_mem=$(awk -v u="$used_gb" -v t="$total_gb" 'BEGIN{printf "%d", u/t*100 + 0.5}')

  printf '\033[H'   # 커서 맨 위 (제자리 덮어쓰기 → 깜빡임 없음)
  printf '%s  %s\033[K\n' "${BOLD}gemma4 모니터${RESET}  $(date '+%Y-%m-%d %H:%M:%S')" "${DIM}(${INTERVAL}s, Ctrl-C 종료)${RESET}"
  printf '%s\033[K\n' "────────────────────────────────────────────────────────────"

  printf '%s\033[K\n' "${BOLD}전체 시스템${RESET}"
  printf 'CPU  '; bar "$sys_cpu" "user+sys (전체 ${ncpu}코어 기준)"; printf '\033[K\n'
  printf 'MEM  '; bar "$sys_mem" "${used_gb} / ${total_gb} GB"; printf '\033[K\n'

  printf '%s\033[K\n' "${BOLD}gemma4 프로세스${RESET}"
  if [ -z "$pid" ]; then
    printf '%s\033[K\n' "서버 미실행 (mlx_vlm.server 프로세스 없음)"
    printf '\033[K\n\033[K\n'
  else
    read -r cpu etime <<< "$(ps -o %cpu=,etime= -p "$pid")"
    cpu_int=$(awk -v c="$cpu" 'BEGIN{printf "%d", c + 0.5}')
    # 실제 점유 메모리: ps RSS는 MLX/Metal 메모리를 못 잡으므로 footprint(phys_footprint) 사용
    fp=$(/usr/bin/footprint -p "$pid" 2>/dev/null | awk '/phys_footprint:/{print $2, $3; exit}')
    mem_gb=$(awk -v s="$fp" 'BEGIN{split(s,a," "); v=a[1]+0; u=a[2]; if(u=="MB")v/=1024; else if(u=="KB")v/=1048576; printf "%.1f", v}')
    mem_pct=$(awk -v m="$mem_gb" -v t="$total_gb" 'BEGIN{printf "%d", (t>0 ? m/t*100 : 0) + 0.5}')
    printf 'CPU  '; bar "$cpu_int" "raw (1코어=100% 기준, 총 ${ncpu}코어)"; printf '\033[K\n'
    printf 'MEM  '; bar "$mem_pct" "${mem_gb} / ${total_gb} GB (footprint)"; printf '\033[K\n'
    printf '%s\033[K\n' "${DIM}PID ${pid} · 실행 ${etime}${RESET}"
  fi

  printf '\033[J'   # 커서 이하 잔여 정리
  sleep "$INTERVAL" & wait "$!"   # & wait 로 두면 시그널(Ctrl+C)에 즉시 반응
done
