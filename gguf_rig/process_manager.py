from __future__ import annotations

import atexit
import json
import os
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
        self._cleanup_stale_process()
        atexit.register(self.shutdown)

    def _append_log(self, message: str) -> None:
        self._log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message.rstrip()}")

    def logs(self, limit: int = 300) -> str:
        with self._lock:
            return "\n".join(list(self._log_lines)[-max(1, int(limit)):])

    def _cleanup_stale_process(self) -> None:
        if not self.config.pid_file.exists():
            return
        try:
            pid = int(self.config.pid_file.read_text(encoding="utf-8").strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            owns_process = "vllm.entrypoints.openai.api_server" in cmdline and str(self.config.models_dir) in cmdline and f"--port {self.config.api_port}" in cmdline
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
        command = self.build_command(active)
        safe_command = ["***" if index and command[index - 1] == "--api-key" else part for index, part in enumerate(command)]
        self._append_log("Starting: " + " ".join(safe_command))
        self._log_handle = self.config.server_log_file.open("a", encoding="utf-8", buffering=1)
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self.config.pid_file.write_text(f"{self._process.pid}\n", encoding="utf-8")
        self._active = active
        self._started_at = time.time()
        threading.Thread(target=self._capture_output, args=(self._process,), daemon=True).start()
        try:
            self._wait_ready()
        except Exception:
            self._stop_locked()
            raise
        self._append_log(f"Ready on {self.config.local_api_url}")
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

    def start(self, active: ActiveModel, *, persist: bool = True) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                raise RuntimeError("vLLM is already running; use switch() or restart()")
            self._intentional_stop = False
            self._launch(active, persist)

    def switch(self, active: ActiveModel) -> str:
        with self._lock:
            previous = self._active if self._process and self._process.poll() is None else None
            if previous == active:
                return "The selected model and dtype are already active."
            self._stop_locked()
            try:
                self._intentional_stop = False
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
            self._launch(active, persist=True)
            return f"Restarted {active.model_id}"

    def restore(self) -> str:
        active = self.load_saved()
        if not active:
            return "No saved model to restore."
        self.start(active, persist=False)
        return f"Restored {active.model_id}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = bool(self._process and self._process.poll() is None)
            healthy, health_detail = self._health() if running else (False, "stopped")
            return {
                "state": "ready" if healthy else ("starting/unhealthy" if running else "stopped"),
                "running": running,
                "healthy": healthy,
                "health_detail": health_detail,
                "pid": self._process.pid if running and self._process else None,
                "model": self._active.model_id if self._active else None,
                "dtype": normalize_dtype(self._active.dtype) if self._active else None,
                "uptime_seconds": int(time.time() - self._started_at) if running and self._started_at else 0,
                "api_url": self.config.local_api_url,
            }

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


# Import compatibility for callers of the previous GGUF implementation.
LlamaServerManager = VllmServerManager
