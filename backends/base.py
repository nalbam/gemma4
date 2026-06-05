"""백엔드 공유 헬퍼."""
import metrics


def idle_process() -> dict:
    """서버 미실행 시 '프로세스/GPU' 패널용 0값."""
    return {"running": False, "pid": None, "cpu": 0, "mem_gb": 0.0,
            "mem_pct": 0, "mem_total_gb": metrics.TOTAL_GB, "etime": ""}
