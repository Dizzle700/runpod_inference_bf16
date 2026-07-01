from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


_gpu_cache_lock = threading.Lock()
_gpu_cache: list[dict[str, Any]] = []
_gpu_cache_time: float = 0.0
_GPU_CACHE_TTL: float = 3.0


def gpu_stats() -> list[dict[str, Any]]:
    """Return GPU info from nvidia-smi, cached for up to 3 seconds."""
    global _gpu_cache, _gpu_cache_time
    now = time.monotonic()
    if now - _gpu_cache_time < _GPU_CACHE_TTL:
        return _gpu_cache
    with _gpu_cache_lock:
        # Double-check after acquiring the lock.
        if now - _gpu_cache_time < _GPU_CACHE_TTL:
            return _gpu_cache
        _gpu_cache = _gpu_stats_uncached()
        _gpu_cache_time = time.monotonic()
        return _gpu_cache


def _gpu_stats_uncached() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, temperature, utilization, used, total = parts
        devices.append(
            {
                "index": int(index),
                "name": name,
                "temperature": int(temperature),
                "utilization": int(utilization),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
            }
        )
    return devices


def disk_stats(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "free_gib": usage.free / 1024**3,
        "used_gib": usage.used / 1024**3,
        "total_gib": usage.total / 1024**3,
    }
