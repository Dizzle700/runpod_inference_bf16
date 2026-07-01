from __future__ import annotations

import atexit
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RigConfig
from .library import ModelLibrary, normalize_dtype


@dataclass(frozen=True)
class ActiveModel:
    model_id: str
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    max_num_seqs: int = 64
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = False
    chat_template: str = ""
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveModel":
        return cls(
            model_id=str(data["model_id"]),
            dtype=normalize_dtype(str(data.get("dtype", "bfloat16"))),
            max_model_len=int(data.get("max_model_len", data.get("context_size", 8192))),
            max_num_seqs=int(data.get("max_num_seqs", 64)),
            tensor_parallel_size=int(data.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.90)),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            chat_template=str(data.get("chat_template", "")),
            enforce_eager=bool(data.get("enforce_eager", False)),
            enable_chunked_prefill=bool(data.get("enable_chunked_prefill", False)),
        )

    def validate(self) -> None:
        normalize_dtype(self.dtype)
        if not 512 <= self.max_model_len <= 1_048_576:
            raise ValueError("Max model length must be between 512 and 1,048,576")
        if not 1 <= self.max_num_seqs <= 4096:
            raise ValueError("Max concurrent sequences must be between 1 and 4,096")
        if not 1 <= self.tensor_parallel_size <= 256:
            raise ValueError("Tensor parallel size must be between 1 and 256")
        if not 0.05 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError("GPU memory utilization must be between 0.05 and 1.0")
        if "\x00" in self.chat_template:
            raise ValueError("Chat template contains an invalid NUL character")


# ---------------------------------------------------------------------------
# Prometheus metrics parser (minimal, for vLLM /metrics endpoint)
# ---------------------------------------------------------------------------

_PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.eE+-]+)$")


def _parse_prometheus_simple(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition text into a flat {metric_name: value} dict.

    Only lines without labels are captured; labeled lines are skipped for
    simplicity.  This is sufficient for the vLLM scalar gauges we care about.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip lines with labels (contain '{').
        if "{" in line:
            continue
        match = _PROM_LINE_RE.match(line)
        if match:
            try:
                result[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    return result


class VllmServerManager:
    """Own exactly one vLLM OpenAI-compatible server process."""

    def __init__(self, config: RigConfig, library: ModelLibrary):
        self.config = config
        self.library = library
        self.config.ensure_directories()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._active: ActiveModel | None = None
        self._started_at: float | None = None
        self._log_lines: deque[str] = deque(maxlen=2_000)
        self._log_handle = None
        self._intentional_stop = False
        # Auto-restart state.
        self._restart_count = 0
        self._last_restart_at: float = 0.0
        # Status cache.
        self._status_cache: dict[str, Any] | None = None
        self._status_cache_time: float = 0.0
        self._STATUS_CACHE_TTL: float = 2.0
        # API statistics (in-memory, reset on app restart).
        self._api_stats_lock = threading.Lock()
        self._api_total_requests: int = 0
        self._api_total_tokens: int = 0
        self._api_total_latency: float = 0.0
        self._api_errors: int = 0

        self._cleanup_stale_process()
        atexit.register(self.shutdown)

    # -- API statistics -------------------------------------------------

    def record_api_call(self, tokens: int = 0, latency: float = 0.0, error: bool = False) -> None:
        with self._api_stats_lock:
            self._api_total_requests += 1
            self._api_total_tokens += tokens
            self._api_total_latency += latency
            if error:
                self._api_errors += 1

    def api_stats(self) -> dict[str, Any]:
        with self._api_stats_lock:
            avg_latency = (self._api_total_latency / self._api_total_requests) if self._api_total_requests else 0.0
            return {
                "total_requests": self._api_total_requests,
                "total_tokens": self._api_total_tokens,
                "avg_latency_s": round(avg_latency, 3),
                "errors": self._api_errors,
            }

    # -- Logging --------------------------------------------------------

    def _append_log(self, message: str) -> None:
        self._log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message.rstrip()}")

    def logs(self, limit: int = 300) -> str:
        with self._lock:
            return "\n".join(list(self._log_lines)[-max(1, int(limit)):])

    # -- Log rotation ---------------------------------------------------

    def _rotate_log_if_needed(self) -> None:
        """Rotate vllm.log when it exceeds max_log_bytes."""
        log_file = self.config.server_log_file
        try:
            if log_file.exists() and log_file.stat().st_size > self.config.max_log_bytes:
                rotated = log_file.with_suffix(".log.1")
                if rotated.exists():
                    rotated.unlink()
                log_file.rename(rotated)
                self._append_log(f"Rotated log file ({self.config.max_log_bytes / 1024 / 1024:.0f} MB limit)")
        except OSError as exc:
            self._append_log(f"Log rotation failed: {exc}")

    # -- Stale process cleanup ------------------------------------------

    def _cleanup_stale_process(self) -> None:
        if not self.config.pid_file.exists():
            return
        try:
            pid = int(self.config.pid_file.read_text(encoding="utf-8").strip())
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            if not cmdline_path.exists():
                return
            raw_args = cmdline_path.read_bytes().split(b"\0")
            args = [arg.decode(errors="replace") for arg in raw_args if arg]
            owns_process = (
                any("vllm.entrypoints.openai.api_server" in arg for arg in args)
                and any(str(self.config.models_dir) in arg for arg in args)
                and str(self.config.api_port) in args
            )
            if owns_process:
                self._append_log(f"Stopping stale managed vLLM process {pid}")
                self._terminate_group(pid)
        except (OSError, ValueError):
            pass
        finally:
            self.config.pid_file.unlink(missing_ok=True)

    def _terminate_group(self, pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + self.config.stop_timeout
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.1)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def build_command(self, active: ActiveModel) -> list[str]:
        active.validate()
        model = self.library.get(active.model_id)
        command = [
            str(self.config.python_executable), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model.path),
            "--served-model-name", "current",
            "--host", self.config.api_host,
            "--port", str(self.config.api_port),
            "--dtype", normalize_dtype(active.dtype),
            "--load-format", "safetensors",
            "--max-model-len", str(active.max_model_len),
            "--max-num-seqs", str(active.max_num_seqs),
            "--tensor-parallel-size", str(active.tensor_parallel_size),
            "--gpu-memory-utilization", str(active.gpu_memory_utilization),
        ]
        if self.config.api_key:
            command.extend(["--api-key", self.config.api_key])
        if active.trust_remote_code:
            command.append("--trust-remote-code")
        if active.chat_template:
            command.extend(["--chat-template", active.chat_template])
        if active.enforce_eager:
            command.append("--enforce-eager")
        if active.enable_chunked_prefill:
            command.append("--enable-chunked-prefill")
        return command

    def _health(self, timeout: float = 1.5) -> tuple[bool, str]:
        request = urllib.request.Request(f"{self.config.local_api_url}/health")
        if self.config.api_key:
            request.add_header("Authorization", f"Bearer {self.config.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 300, response.read(2048).decode(errors="replace")
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except (OSError, TimeoutError) as exc:
            return False, str(exc)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.health_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            if self._process is None:
                raise RuntimeError("vLLM process disappeared")
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(f"vLLM exited during startup with code {return_code}\n{self.logs(40)}")
            healthy, last_error = self._health()
            if healthy:
                return
            time.sleep(1)
        raise TimeoutError(f"vLLM did not become ready in {self.config.health_timeout}s: {last_error}")

    def _write_state(self, active: ActiveModel) -> None:
        payload = {"schema_version": 2, **asdict(active), "updated_at": int(time.time())}
        fd, temp_name = tempfile.mkstemp(prefix="active-model-", suffix=".json", dir=self.config.state_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.config.active_model_file)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def load_saved(self) -> ActiveModel | None:
        try:
            active = ActiveModel.from_dict(json.loads(self.config.active_model_file.read_text(encoding="utf-8")))
            self.library.get(active.model_id)
            active.validate()
            return active
        except FileNotFoundError:
            return None
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            self._append_log(f"Ignoring invalid saved state: {exc}")
            return None

    def _launch(self, active: ActiveModel, persist: bool) -> None:
        if not self.config.python_executable.is_file():
            raise FileNotFoundError(f"Python executable not found at {self.config.python_executable}")
        self._rotate_log_if_needed()
        command = self.build_command(active)
        safe_command = ["***" if index and command[index - 1] == "--api-key" else part for index, part in enumerate(command)]
        self._append_log("Starting: " + " ".join(safe_command))
        self._log_handle = self.config.server_log_file.open("a", encoding="utf-8", buffering=1)
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self.config.pid_file.write_text(f"{self._process.pid}\n", encoding="utf-8")
        self._active = active
        self._started_at = time.time()
        self._invalidate_status_cache()
        threading.Thread(target=self._capture_output, args=(self._process,), daemon=True).start()
        try:
            self._wait_ready()
        except Exception:
            self._stop_locked()
            raise
        self._append_log(f"Ready on {self.config.local_api_url}")
        self._restart_count = 0  # Reset restart counter on success.
        if persist:
            self._write_state(active)

    def _capture_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            clean = line.rstrip()
            if clean:
                self._append_log(clean)
                try:
                    if self._log_handle and not self._log_handle.closed:
                        self._log_handle.write(clean + "\n")
                except (OSError, ValueError):
                    pass
        return_code = process.wait()
        if process is self._process and not self._intentional_stop:
            self._append_log(f"vLLM exited unexpectedly with code {return_code}")
            self._invalidate_status_cache()
            if self.config.auto_restart:
                self._try_auto_restart()

    def _try_auto_restart(self) -> None:
        """Attempt automatic restart with exponential backoff."""
        with self._lock:
            if self._intentional_stop:
                return
            active = self._active or self.load_saved()
            if not active:
                self._append_log("Auto-restart: no model to restart")
                return
            if self._restart_count >= self.config.auto_restart_max_retries:
                self._append_log(
                    f"Auto-restart: giving up after {self._restart_count} failed attempts"
                )
                return
            self._restart_count += 1
            backoff = min(2 ** (self._restart_count - 1), 30)
            self._append_log(
                f"Auto-restart: attempt {self._restart_count}/{self.config.auto_restart_max_retries} "
                f"in {backoff}s"
            )

        time.sleep(backoff)

        with self._lock:
            if self._intentional_stop:
                return
            # Clean up the dead process before relaunch.
            self._process = None
            self._started_at = None
            self.config.pid_file.unlink(missing_ok=True)
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
            try:
                self._intentional_stop = False
                self._launch(active, persist=False)
                self._append_log("Auto-restart: success")
            except Exception as exc:
                self._append_log(f"Auto-restart: failed — {exc}")

    def start(self, active: ActiveModel, *, persist: bool = True) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                raise RuntimeError("vLLM is already running; use switch() or restart()")
            self._intentional_stop = False
            self._restart_count = 0
            self._launch(active, persist)

    def switch(self, active: ActiveModel) -> str:
        with self._lock:
            previous = self._active if self._process and self._process.poll() is None else None
            if previous == active:
                return "The selected model and dtype are already active."
            self._stop_locked()
            try:
                self._intentional_stop = False
                self._restart_count = 0
                self._launch(active, persist=True)
                return f"Activated {active.model_id} as {normalize_dtype(active.dtype)}"
            except Exception as new_error:
                self._append_log(f"Activation failed: {new_error}")
                if previous:
                    try:
                        self._intentional_stop = False
                        self._launch(previous, persist=False)
                    except Exception as rollback_error:
                        raise RuntimeError(f"New model failed: {new_error}; rollback also failed: {rollback_error}") from new_error
                    raise RuntimeError(f"New model failed; rolled back to {previous.model_id}: {new_error}") from new_error
                raise

    def _stop_locked(self) -> None:
        process = self._process
        if process is None:
            self.config.pid_file.unlink(missing_ok=True)
            return
        self._intentional_stop = True
        if process.poll() is None:
            self._append_log(f"Sending SIGTERM to process group {process.pid}")
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self.config.stop_timeout)
            except subprocess.TimeoutExpired:
                self._append_log("Grace period expired; sending SIGKILL")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            except ProcessLookupError:
                pass
        self._process = None
        self._started_at = None
        self.config.pid_file.unlink(missing_ok=True)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
        self._invalidate_status_cache()

    def stop(self) -> str:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                self._process = None
                return "vLLM is already stopped."
            self._stop_locked()
            return "vLLM stopped."

    def restart(self) -> str:
        with self._lock:
            active = self._active or self.load_saved()
            if not active:
                raise RuntimeError("No active or saved model to restart")
            self._stop_locked()
            self._intentional_stop = False
            self._restart_count = 0
            self._launch(active, persist=True)
            return f"Restarted {active.model_id}"

    def restore(self) -> str:
        active = self.load_saved()
        if not active:
            return "No saved model to restore."
        self.start(active, persist=False)
        return f"Restored {active.model_id}"

    # -- Status with TTL cache ------------------------------------------

    def _invalidate_status_cache(self) -> None:
        self._status_cache = None
        self._status_cache_time = 0.0

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._status_cache is not None and now - self._status_cache_time < self._STATUS_CACHE_TTL:
            return self._status_cache
        with self._lock:
            running = bool(self._process and self._process.poll() is None)
            healthy, health_detail = self._health() if running else (False, "stopped")
            result = {
                "state": "ready" if healthy else ("starting/unhealthy" if running else "stopped"),
                "running": running,
                "healthy": healthy,
                "health_detail": health_detail,
                "pid": self._process.pid if running and self._process else None,
                "model": self._active.model_id if self._active else None,
                "dtype": normalize_dtype(self._active.dtype) if self._active else None,
                "uptime_seconds": int(time.time() - self._started_at) if running and self._started_at else 0,
                "api_url": self.config.local_api_url,
                # Extended info for dashboard.
                "max_model_len": self._active.max_model_len if self._active else None,
                "max_num_seqs": self._active.max_num_seqs if self._active else None,
                "tensor_parallel_size": self._active.tensor_parallel_size if self._active else None,
                "gpu_memory_utilization": self._active.gpu_memory_utilization if self._active else None,
                "enforce_eager": self._active.enforce_eager if self._active else None,
                "enable_chunked_prefill": self._active.enable_chunked_prefill if self._active else None,
                "auto_restart": self.config.auto_restart,
            }
            self._status_cache = result
            self._status_cache_time = time.monotonic()
            return result

    # -- vLLM Prometheus metrics ----------------------------------------

    def metrics(self) -> dict[str, Any]:
        """Fetch key metrics from vLLM /metrics (Prometheus) endpoint."""
        result: dict[str, Any] = {
            "active_requests": None,
            "pending_requests": None,
            "kv_cache_usage_percent": None,
            "generation_tokens_total": None,
            "prompt_tokens_total": None,
            "avg_generation_throughput": None,
        }
        try:
            req = urllib.request.Request(f"{self.config.local_api_url}/metrics")
            with urllib.request.urlopen(req, timeout=2) as resp:
                text = resp.read().decode(errors="replace")
        except Exception:
            return result

        data = _parse_prometheus_simple(text)

        # Map well-known vLLM metric names to our result keys.
        result["active_requests"] = data.get(
            "vllm:num_requests_running",
            data.get("vllm_num_requests_running"),
        )
        result["pending_requests"] = data.get(
            "vllm:num_requests_waiting",
            data.get("vllm_num_requests_waiting"),
        )
        result["kv_cache_usage_percent"] = data.get(
            "vllm:gpu_cache_usage_perc",
            data.get("vllm_gpu_cache_usage_perc"),
        )
        result["generation_tokens_total"] = data.get(
            "vllm:generation_tokens_total",
            data.get("vllm_generation_tokens_total"),
        )
        result["prompt_tokens_total"] = data.get(
            "vllm:prompt_tokens_total",
            data.get("vllm_prompt_tokens_total"),
        )
        result["avg_generation_throughput"] = data.get(
            "vllm:avg_generation_throughput_toks_per_s",
            data.get("vllm_avg_generation_throughput_toks_per_s"),
        )
        return result

    # -- Served model name accessor -------------------------------------

    def served_model_name(self) -> str:
        """Return the --served-model-name used by the running vLLM."""
        return "current"

    # -- Shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
