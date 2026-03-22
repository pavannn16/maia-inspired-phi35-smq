from __future__ import annotations

import dataclasses
import json
import os
import platform
import time
from typing import Any, Dict, Optional

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


def now_ms() -> float:
    return time.time() * 1000.0


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if orjson is not None:
        line = orjson.dumps(record) + b"\n"
    else:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "ab") as f:
        f.write(line)


def get_env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    try:
        import torch

        snap.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            snap.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_cc": torch.cuda.get_device_capability(0),
                    "gpu_mem_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                }
            )
    except Exception as e:
        snap["torch_error"] = str(e)

    for k in ["COLAB_RELEASE_TAG", "NVIDIA_VISIBLE_DEVICES"]:
        if k in os.environ:
            snap[k] = os.environ[k]

    return snap


def maybe_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU state snapshot (clock speed + temperature — flags throttled sessions)
# ---------------------------------------------------------------------------

def get_gpu_state() -> Dict[str, Any]:
    """Record SM clock, memory clock, temperature, and throttle reason.

    A temperature > 75 °C or non-zero throttle reason should be flagged in
    the analysis as a potentially degraded measurement.
    """
    state: Dict[str, Any] = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        state["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        state["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        state["temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        state["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        state["throttle_reasons"] = str(
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        )
    except Exception as e:
        state["pynvml_error"] = str(e)
    return state


# ---------------------------------------------------------------------------
# Power sampler — background thread for energy measurement
# ---------------------------------------------------------------------------

import threading
from typing import List, Tuple


class PowerSampler:
    """Background thread that samples GPU power via pynvml at ~100 ms intervals.

    Integrates power over time (trapezoidal rule) to yield joules consumed.
    On Colab, pynvml sampling may be restricted; the sampler degrades
    gracefully and returns None when unavailable.

    Usage::

        sampler = PowerSampler()
        sampler.start()
        # ... timed workload ...
        joules = sampler.stop()   # None if pynvml unavailable
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.1) -> None:
        self._idx = device_index
        self._interval = interval_s
        self._samples: List[Tuple[float, float]] = []  # (timestamp_s, watts)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle: Optional[Any] = None

    def start(self) -> bool:
        """Begin sampling.  Returns True if pynvml is functional."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._idx)
        except Exception:
            return False
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        import pynvml
        while self._running:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append((time.time(), mw / 1000.0))
            except Exception:
                pass
            time.sleep(self._interval)

    def stop(self) -> Optional[float]:
        """Stop sampling and return joules (trapezoidal integration), or None."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if len(self._samples) < 2:
            return None
        ts = [s[0] for s in self._samples]
        ws = [s[1] for s in self._samples]
        return sum(
            (ws[i] + ws[i + 1]) * 0.5 * (ts[i + 1] - ts[i])
            for i in range(len(ts) - 1)
        )
