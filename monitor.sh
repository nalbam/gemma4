#!/usr/bin/env bash
# gemma4 자원 모니터링 — 전체 시스템 / 가속기의 CPU·MEM 막대 그래프.
# macOS(MLX, footprint) / Linux(Ollama + NVIDIA GPU) 를 자동 분기한다.
# 인자: 갱신 간격(초, 기본 2)
set -uo pipefail

INTERVAL="${1:-2}"
WIDTH=30
OS="$(uname)"

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

# ---- 플랫폼별 1회 설정 ----
if [ "$OS" = "Darwin" ]; then
  total_bytes=$(sysctl -n hw.memsize)
  total_gb=$(awk -v b="$total_bytes" 'BEGIN{printf "%.1f", b/1073741824}')
  ncpu=$(sysctl -n hw.ncpu)
  PROC_TITLE="gemma4 프로세스"
else
  total_kb=$(awk '/^MemTotal/{print $2; exit}' /proc/meminfo)
  total_gb=$(awk -v k="$total_kb" 'BEGIN{printf "%.1f", k*1024/1073741824}')
  ncpu=$(nproc)
  PROC_TITLE="gemma4 GPU (NVIDIA)"
fi

prev_idle=0; prev_total=0   # Linux CPU delta 용

# 시스템 수집 → sys_cpu, used_gb 설정
collect_sys_mac() {
  local topout; topout="$(top -l1 -n0)"
  sys_cpu=$(awk '/CPU usage/{v=$7; gsub(/%/,"",v); printf "%d", 100-v+0.5}' <<< "$topout")
  used_gb=$(vm_stat | awk '
    /page size of/ { match($0, /[0-9]+/); ps = substr($0, RSTART, RLENGTH) }
    /Pages wired down/ { gsub(/\./, ""); w = $NF }
    /Anonymous pages/ { gsub(/\./, ""); a = $NF }
    /Pages occupied by compressor/ { gsub(/\./, ""); c = $NF }
    END { printf "%.1f", (w + a + c) * ps / 1073741824 }')
}
collect_sys_linux() {
  set -- $(awk '/^cpu /{for (i = 2; i <= NF; i++) printf "%s ", $i; exit}' /proc/stat)
  local idle=$(( $4 + ${5:-0} )) total=0 v   # user nice system idle iowait ...
  for v in "$@"; do total=$((total + v)); done
  local di=$((idle - prev_idle)) dt=$((total - prev_total))
  if (( dt > 0 )); then sys_cpu=$(( (dt - di) * 100 / dt )); else sys_cpu=0; fi
  prev_idle=$idle; prev_total=$total
  used_gb=$(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} END{printf "%.1f", (t-a)*1024/1073741824}' /proc/meminfo)
}

# 가속기 수집 → proc_running, p_cpu, p_cpu_suffix, p_mem_gb, p_mem_total, p_mem_pct, p_info
collect_proc_mac() {
  local pid; pid="$(pgrep -f mlx_vlm.server | head -1 || true)"
  if [ -z "$pid" ]; then proc_running=0; return; fi
  proc_running=1
  local cpu etime; read -r cpu etime <<< "$(ps -o %cpu=,etime= -p "$pid")"
  p_cpu=$(awk -v c="$cpu" 'BEGIN{printf "%d", c + 0.5}')
  p_cpu_suffix="raw (1코어=100% 기준, 총 ${ncpu}코어)"
  local fp; fp=$(/usr/bin/footprint -p "$pid" 2>/dev/null | awk '/phys_footprint:/{print $2, $3; exit}')
  p_mem_gb=$(awk -v s="$fp" 'BEGIN{split(s,a," "); v=a[1]+0; u=a[2]; if(u=="MB")v/=1024; else if(u=="KB")v/=1048576; printf "%.1f", v}')
  p_mem_total=$total_gb
  p_mem_pct=$(awk -v m="$p_mem_gb" -v t="$total_gb" 'BEGIN{printf "%d", (t>0 ? m/t*100 : 0) + 0.5}')
  p_info="PID ${pid} · 실행 ${etime}"
}
collect_proc_linux() {
  if ! curl -s -o /dev/null http://127.0.0.1:11434/api/tags; then proc_running=0; return; fi
  proc_running=1
  local util used total; read -r util used total <<< "$(nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr ',' ' ')"
  p_cpu=${util:-0}
  p_cpu_suffix="GPU util"
  p_mem_gb=$(awk -v u="${used:-0}" 'BEGIN{printf "%.1f", u/1024}')
  p_mem_total=$(awk -v t="${total:-0}" 'BEGIN{printf "%.1f", t/1024}')
  p_mem_pct=$(awk -v u="${used:-0}" -v t="${total:-1}" 'BEGIN{printf "%d", (t>0 ? u/t*100 : 0) + 0.5}')
  local gname; gname=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  p_info="${gname} · VRAM ${p_mem_gb}/${p_mem_total} GB"
}

trap 'printf "\033[?25h\n"' EXIT      # 종료 시 커서 복원
trap 'exit 130' INT TERM             # Ctrl+C/TERM → 실제 종료
printf '\033[2J\033[?25l'            # 전체 클리어 1회 + 커서 숨김

while true; do
  if [ "$OS" = "Darwin" ]; then collect_sys_mac; collect_proc_mac; else collect_sys_linux; collect_proc_linux; fi
  sys_mem=$(awk -v u="$used_gb" -v t="$total_gb" 'BEGIN{printf "%d", u/t*100 + 0.5}')

  printf '\033[H'   # 커서 맨 위 (제자리 덮어쓰기 → 깜빡임 없음)
  printf '%s  %s\033[K\n' "${BOLD}gemma4 모니터${RESET}  $(date '+%Y-%m-%d %H:%M:%S')" "${DIM}(${INTERVAL}s, Ctrl-C 종료)${RESET}"
  printf '%s\033[K\n' "────────────────────────────────────────────────────────────"

  printf '%s\033[K\n' "${BOLD}전체 시스템${RESET}"
  printf 'CPU  '; bar "$sys_cpu" "user+sys (전체 ${ncpu}코어 기준)"; printf '\033[K\n'
  printf 'MEM  '; bar "$sys_mem" "${used_gb} / ${total_gb} GB"; printf '\033[K\n'

  printf '%s\033[K\n' "${BOLD}${PROC_TITLE}${RESET}"
  if [ "${proc_running}" = "0" ]; then
    if [ "$OS" = "Darwin" ]; then
      printf '%s\033[K\n' "서버 미실행 (mlx_vlm.server 프로세스 없음)"
    else
      printf '%s\033[K\n' "ollama 데몬 미응답 (systemctl status ollama)"
    fi
    printf '\033[K\n\033[K\n'
  else
    printf 'CPU  '; bar "$p_cpu" "$p_cpu_suffix"; printf '\033[K\n'
    printf 'MEM  '; bar "$p_mem_pct" "${p_mem_gb} / ${p_mem_total} GB"; printf '\033[K\n'
    printf '%s\033[K\n' "${DIM}${p_info}${RESET}"
  fi

  printf '\033[J'   # 커서 이하 잔여 정리
  sleep "$INTERVAL" & wait "$!"
done
