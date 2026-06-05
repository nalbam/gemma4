"""시스템 자원 메트릭 — macOS / Linux 를 런타임에 분기한다.

전체 시스템 CPU·MEM 만 책임진다. 서빙 프로세스/가속기(GPU) 메트릭은 각 backend 가
`process_metrics()` 로 제공한다 (backend.py).
"""
import platform
import re
import subprocess
import time
from pathlib import Path

IS_MAC = platform.system() == "Darwin"


def _total_bytes_mac() -> int:
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())


def _total_bytes_linux() -> int:
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text())
    return int(m.group(1)) * 1024 if m else 0


def _ncpu_mac() -> int:
    return int(subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip())


def _ncpu_linux() -> int:
    import os
    return os.cpu_count() or 1


TOTAL_BYTES = _total_bytes_mac() if IS_MAC else _total_bytes_linux()
TOTAL_GB = round(TOTAL_BYTES / 1073741824, 1)
NCPU = _ncpu_mac() if IS_MAC else _ncpu_linux()


# ---- macOS ----
def _used_mem_gb_mac() -> float:
    """실제 사용 메모리 = anonymous + wired + compressed.
    top의 PhysMem 'used'는 inactive·파일 캐시(회수 가능)까지 포함해 과대 표시되므로 쓰지 않는다."""
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    psize = 4096
    m = re.search(r"page size of (\d+)", vm)
    if m:
        psize = int(m.group(1))

    def pages(name: str) -> int:
        mm = re.search(rf"{re.escape(name)}:\s+(\d+)", vm)
        return int(mm.group(1)) if mm else 0

    used_pages = pages("Pages wired down") + pages("Anonymous pages") + pages("Pages occupied by compressor")
    return round(used_pages * psize / 1073741824, 1)


def _system_mac() -> dict:
    out = subprocess.run(["top", "-l1", "-n0"], capture_output=True, text=True).stdout
    cpu = 0
    for line in out.splitlines():
        if "CPU usage" in line:
            m = re.search(r"([\d.]+)%\s+idle", line)
            if m:
                cpu = round(100 - float(m.group(1)))
            break
    mem_used_gb = _used_mem_gb_mac()
    return {
        "cpu": cpu,
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": TOTAL_GB,
        "mem_pct": round(mem_used_gb / TOTAL_GB * 100) if TOTAL_GB else 0,
    }


# ---- Linux ----
def _cpu_linux() -> int:
    """/proc/stat 두 샘플 사이의 busy 비율 (0~100). 0.1초 간격."""
    def snap() -> tuple[int, int]:
        parts = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
        return idle, sum(parts)

    i1, t1 = snap()
    time.sleep(0.1)
    i2, t2 = snap()
    dt, di = t2 - t1, i2 - i1
    return round((1 - di / dt) * 100) if dt else 0


def _used_mem_gb_linux() -> float:
    info = Path("/proc/meminfo").read_text()
    total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1))
    avail = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1))
    return round((total - avail) * 1024 / 1073741824, 1)


def _system_linux() -> dict:
    mem_used_gb = _used_mem_gb_linux()
    return {
        "cpu": _cpu_linux(),
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": TOTAL_GB,
        "mem_pct": round(mem_used_gb / TOTAL_GB * 100) if TOTAL_GB else 0,
    }


def system() -> dict:
    """전체 시스템 CPU(%) + 실제 사용 메모리."""
    return _system_mac() if IS_MAC else _system_linux()
