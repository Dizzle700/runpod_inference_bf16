from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, legacy_name: str, default: str | Path) -> str:
    return os.environ.get(name, os.environ.get(legacy_name, str(default)))


def _env_bool(name: str, legacy_name: str = "", default: bool = False) -> bool:
    value = os.environ.get(name, os.environ.get(legacy_name) if legacy_name else None)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RigConfig:
    volume_root: Path
    models_dir: Path
    state_dir: Path
    log_dir: Path
    python_executable: Path
    api_host: str
    api_port: int
    panel_host: str
    panel_port: int
    api_key: str
    panel_user: str
    panel_password: str
    hf_token: str
    allow_insecure: bool
    health_timeout: int
    stop_timeout: int
    auto_restart: bool = False
    auto_restart_max_retries: int = 3
    max_log_bytes: int = 50 * 1024 * 1024  # 50 MB
    panel_find_free_port: bool = False
    max_audio_upload_bytes: int = 100 * 1024 * 1024
    bootstrap_model: str = ""
    bootstrap_revision: str = "main"
    bootstrap_dtype: str = "bfloat16"
    bootstrap_max_model_len: int = 8192
    bootstrap_max_num_seqs: int = 64
    bootstrap_tensor_parallel_size: int = 1
    bootstrap_gpu_memory_utilization: float = 0.90
    bootstrap_trust_remote_code: bool = False
    bootstrap_enforce_eager: bool = False
    bootstrap_enable_chunked_prefill: bool = False
    bootstrap_auto_adjust_context: bool = True
    bootstrap_max_audio_per_prompt: int = 1
    public_api_url_override: str = ""
    public_panel_url_override: str = ""

    @classmethod
    def from_env(cls) -> "RigConfig":
        project_dir = Path(__file__).resolve().parents[1]
        default_volume = Path("/workspace") if Path("/workspace").is_dir() else project_dir / "data"
        volume_root = Path(_env("SAFETENSORS_VOLUME_ROOT", "GGUF_VOLUME_ROOT", default_volume)).expanduser().resolve()
        return cls(
            volume_root=volume_root,
            models_dir=Path(_env("SAFETENSORS_MODELS_DIR", "GGUF_MODELS_DIR", volume_root / "models" / "safetensors")).expanduser(),
            state_dir=Path(_env("SAFETENSORS_STATE_DIR", "GGUF_STATE_DIR", volume_root / ".state" / "safetensors-rig")).expanduser(),
            log_dir=Path(_env("SAFETENSORS_LOG_DIR", "GGUF_LOG_DIR", volume_root / "logs" / "safetensors-rig")).expanduser(),
            python_executable=Path(os.environ.get("VLLM_PYTHON", sys.executable)).expanduser(),
            api_host=_env("SAFETENSORS_API_HOST", "GGUF_API_HOST", "0.0.0.0"),
            api_port=int(_env("SAFETENSORS_API_PORT", "GGUF_API_PORT", "8000")),
            panel_host=_env("SAFETENSORS_PANEL_HOST", "GGUF_PANEL_HOST", "0.0.0.0"),
            panel_port=int(_env("SAFETENSORS_PANEL_PORT", "GGUF_PANEL_PORT", "7860")),
            api_key=_env("SAFETENSORS_API_KEY", "GGUF_API_KEY", ""),
            panel_user=_env("SAFETENSORS_PANEL_USER", "GGUF_PANEL_USER", os.environ.get("PANEL_USER", "")),
            panel_password=_env("SAFETENSORS_PANEL_PASSWORD", "GGUF_PANEL_PASSWORD", os.environ.get("PANEL_PASS", "")),
            hf_token=os.environ.get("HF_TOKEN", ""),
            allow_insecure=_env_bool("SAFETENSORS_ALLOW_INSECURE", "GGUF_ALLOW_INSECURE"),
            health_timeout=int(_env("SAFETENSORS_HEALTH_TIMEOUT", "GGUF_HEALTH_TIMEOUT", "600")),
            stop_timeout=int(_env("SAFETENSORS_STOP_TIMEOUT", "GGUF_STOP_TIMEOUT", "30")),
            auto_restart=_env_bool("SAFETENSORS_AUTO_RESTART"),
            auto_restart_max_retries=int(os.environ.get("SAFETENSORS_AUTO_RESTART_MAX_RETRIES", "3")),
            max_log_bytes=int(os.environ.get("SAFETENSORS_MAX_LOG_BYTES", str(50 * 1024 * 1024))),
            panel_find_free_port=_env_bool("SAFETENSORS_PANEL_FIND_FREE_PORT"),
            max_audio_upload_bytes=int(
                os.environ.get("SAFETENSORS_MAX_AUDIO_UPLOAD_BYTES", str(100 * 1024 * 1024))
            ),
            bootstrap_model=os.environ.get("SAFETENSORS_BOOTSTRAP_MODEL", "").strip(),
            bootstrap_revision=os.environ.get("SAFETENSORS_BOOTSTRAP_REVISION", "main").strip() or "main",
            bootstrap_dtype=os.environ.get("SAFETENSORS_BOOTSTRAP_DTYPE", "bfloat16").strip(),
            bootstrap_max_model_len=int(os.environ.get("SAFETENSORS_BOOTSTRAP_MAX_MODEL_LEN", "8192")),
            bootstrap_max_num_seqs=int(os.environ.get("SAFETENSORS_BOOTSTRAP_MAX_NUM_SEQS", "64")),
            bootstrap_tensor_parallel_size=int(
                os.environ.get("SAFETENSORS_BOOTSTRAP_TENSOR_PARALLEL_SIZE", "1")
            ),
            bootstrap_gpu_memory_utilization=float(
                os.environ.get("SAFETENSORS_BOOTSTRAP_GPU_MEMORY_UTILIZATION", "0.90")
            ),
            bootstrap_trust_remote_code=_env_bool("SAFETENSORS_BOOTSTRAP_TRUST_REMOTE_CODE"),
            bootstrap_enforce_eager=_env_bool("SAFETENSORS_BOOTSTRAP_ENFORCE_EAGER"),
            bootstrap_enable_chunked_prefill=_env_bool("SAFETENSORS_BOOTSTRAP_ENABLE_CHUNKED_PREFILL"),
            bootstrap_auto_adjust_context=_env_bool(
                "SAFETENSORS_BOOTSTRAP_AUTO_ADJUST_CONTEXT", default=True
            ),
            bootstrap_max_audio_per_prompt=int(
                os.environ.get("SAFETENSORS_BOOTSTRAP_MAX_AUDIO_PER_PROMPT", "1")
            ),
            public_api_url_override=os.environ.get("SAFETENSORS_PUBLIC_API_URL", "").strip(),
            public_panel_url_override=os.environ.get("SAFETENSORS_PUBLIC_PANEL_URL", "").strip(),
        )

    @property
    def active_model_file(self) -> Path:
        return self.state_dir / "active_model.json"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "vllm.pid"

    @property
    def server_log_file(self) -> Path:
        return self.log_dir / "vllm.log"

    @property
    def event_log_file(self) -> Path:
        return self.log_dir / "events.jsonl"

    @property
    def local_api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def _runpod_proxy_url(self, port: int, override: str) -> str:
        """Return a public URL when running in a RunPod Pod.

        An explicit override also supports custom domains and other hosting
        providers. RunPod exposes HTTP ports using this predictable proxy URL.
        """
        if override:
            return override.rstrip("/")
        pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
        if not pod_id:
            return ""
        return f"https://{pod_id}-{port}.proxy.runpod.net"

    @property
    def public_api_url(self) -> str:
        return self._runpod_proxy_url(self.api_port, self.public_api_url_override)

    @property
    def public_panel_url(self) -> str:
        return self._runpod_proxy_url(self.panel_port, self.public_panel_url_override)

    def ensure_directories(self) -> None:
        for path in (self.models_dir, self.state_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def validate_security(self) -> None:
        if self.allow_insecure:
            return
        errors: list[str] = []
        if self.api_host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key:
            errors.append("SAFETENSORS_API_KEY is required when the API listens publicly")
        if self.panel_host not in {"127.0.0.1", "localhost", "::1"} and (not self.panel_user or not self.panel_password):
            errors.append("SAFETENSORS_PANEL_USER and SAFETENSORS_PANEL_PASSWORD are required for a public panel")
        if errors:
            raise RuntimeError("; ".join(errors) + ". Set SAFETENSORS_ALLOW_INSECURE=1 only for trusted local development.")
