from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .config import RigConfig


REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
DTYPE_NAMES = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}
IGNORED_WEIGHT_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack", "*.onnx", "*.gguf"]


@dataclass(frozen=True)
class ModelRecord:
    id: str
    path: Path
    size_bytes: int
    dtypes: tuple[str, ...]
    shard_count: int

    @property
    def size_gib(self) -> float:
        return self.size_bytes / 1024**3

    @property
    def dtype_label(self) -> str:
        return "/".join(self.dtypes) if self.dtypes else "unknown"


@dataclass(frozen=True)
class RemoteModel:
    repo_id: str
    size_bytes: int | None
    shard_count: int
    config_dtype: str

    @property
    def size_label(self) -> str:
        return "unknown" if self.size_bytes is None else f"{self.size_bytes / 1024**3:.2f} GiB"


def _read_safetensors_dtypes(path: Path) -> set[str]:
    """Read only the small JSON header; tensor data is never loaded into RAM."""
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return set()
            header_length = int.from_bytes(raw_length, "little")
            if not 2 <= header_length <= 128 * 1024**2:
                return set()
            header = json.loads(handle.read(header_length))
        return {
            DTYPE_NAMES.get(str(value.get("dtype", "")).upper(), str(value.get("dtype", "")).lower())
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict) and value.get("dtype")
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return set()


def normalize_dtype(value: str) -> str:
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "f16": "float16",
        "half": "float16",
        "float16": "float16",
        "fp32": "float32",
        "f32": "float32",
        "float": "float32",
        "float32": "float32",
    }
    try:
        return aliases[value.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("dtype must be one of: bf16, fp16, fp32") from exc


class ModelLibrary:
    def __init__(self, config: RigConfig):
        self.config = config
        self.config.ensure_directories()

    def _safe_local_path(self, path: Path | str) -> Path:
        root = self.config.models_dir.resolve()
        candidate = Path(path).expanduser().resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Path escapes the model library: {candidate}")
        return candidate

    @staticmethod
    def validate_repo_id(repo_id: str) -> str:
        value = repo_id.strip()
        if value.startswith("hf://"):
            value = value.removeprefix("hf://")
        parsed = urlparse(value)
        if parsed.scheme or parsed.netloc:
            if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
                raise ValueError("Only huggingface.co repository URLs are supported")
            parts = [part for part in parsed.path.split("/") if part]
            if parts and parts[0] == "models":
                parts = parts[1:]
            if len(parts) < 2 or parts[0] in {"datasets", "spaces"}:
                raise ValueError("The URL must point to a Hugging Face model repository")
            value = "/".join(parts[:2])
        value = value.strip("/")
        if value.endswith(".git"):
            value = value[:-4]
        if not REPO_RE.fullmatch(value):
            raise ValueError("Use organization/name or a huggingface.co model URL")
        return value

    def scan(self) -> list[ModelRecord]:
        root = self.config.models_dir.resolve()
        records: list[ModelRecord] = []
        for config_path in root.rglob("config.json"):
            model_dir = config_path.parent
            if ".cache" in model_dir.parts:
                continue
            shards = sorted(model_dir.glob("*.safetensors"))
            if not shards:
                continue
            dtypes: set[str] = set()
            for shard in shards:
                dtypes.update(_read_safetensors_dtypes(shard))
            records.append(ModelRecord(
                id=model_dir.relative_to(root).as_posix(),
                path=model_dir,
                size_bytes=sum(shard.stat().st_size for shard in shards),
                dtypes=tuple(sorted(dtypes)),
                shard_count=len(shards),
            ))
        return sorted(records, key=lambda item: item.id.lower())

    def get(self, model_id: str) -> ModelRecord:
        model_dir = self._safe_local_path(self.config.models_dir / model_id)
        if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Model not found: {model_id}")
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No Safetensors weights found in: {model_id}")
        dtypes: set[str] = set()
        for shard in shards:
            dtypes.update(_read_safetensors_dtypes(shard))
        return ModelRecord(model_dir.relative_to(self.config.models_dir.resolve()).as_posix(), model_dir, sum(p.stat().st_size for p in shards), tuple(sorted(dtypes)), len(shards))

    def inspect_remote(self, repo_id: str, token: str | None = None) -> RemoteModel:
        repo_id = self.validate_repo_id(repo_id)
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is not installed") from exc
        info = HfApi(token=token or None).model_info(repo_id=repo_id, files_metadata=True)
        tensors = [item for item in info.siblings or [] if getattr(item, "rfilename", "").lower().endswith(".safetensors")]
        if not tensors:
            raise ValueError("This repository has no .safetensors weights")
        sizes = [getattr(item, "size", None) for item in tensors]
        total = sum(sizes) if all(size is not None for size in sizes) else None
        config_dtype = "unknown"
        safetensors_meta = getattr(info, "safetensors", None)
        parameters = getattr(safetensors_meta, "parameters", None)
        if isinstance(parameters, dict) and parameters:
            config_dtype = "/".join(sorted(str(key).lower() for key in parameters))
        return RemoteModel(repo_id, total, len(tensors), config_dtype)

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.config.models_dir).free

    def download_snapshot(self, repo_id: str, *, token: str | None = None, expected_size: int | None = None, progress: Callable[[float, str], None] | None = None) -> Path:
        repo_id = self.validate_repo_id(repo_id)
        if expected_size is not None and self.free_bytes() < expected_size + 1024**3:
            raise OSError("Not enough free volume space (a 1 GiB safety margin is required)")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is not installed") from exc
        destination = self.config.models_dir.joinpath(*repo_id.split("/"))
        destination.mkdir(parents=True, exist_ok=True)

        class TqdmProgressWrapper:
            def __init__(self, *args, **kwargs):
                self._total = kwargs.get("total") or 100
                self._n = 0
                self._desc = kwargs.get("desc") or "Downloading"
                self._unit = kwargs.get("unit") or "B"
                if progress:
                    progress(0.0, self._desc)

            def update(self, n=1):
                self._n += n
                ratio = min(1.0, max(0.0, self._n / self._total))
                if progress:
                    if self._unit.lower() in ("b", "byte", "bytes"):
                        desc = f"{self._desc} ({self._n / 1024**2:.1f}MB / {self._total / 1024**2:.1f}MB)"
                    else:
                        desc = f"{self._desc} ({self._n} / {self._total})"
                    progress(ratio, desc)

            def close(self):
                if progress:
                    progress(1.0, self._desc)

            def set_description(self, desc, refresh=True):
                self._desc = desc

            def set_postfix(self, *args, **kwargs):
                pass

            def refresh(self):
                pass

            def reset(self, total=None):
                if total is not None:
                    self._total = total
                self._n = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()

        class PatchTqdm:
            def __init__(self, wrapper):
                self.wrapper = wrapper
                self.originals = {}

            def __enter__(self):
                import sys
                import tqdm
                import tqdm.auto
                for mod in (tqdm, tqdm.auto):
                    if hasattr(mod, "tqdm"):
                        self.originals[(mod, "tqdm")] = getattr(mod, "tqdm")
                        setattr(mod, "tqdm", self.wrapper)
                for name, module in list(sys.modules).items():
                    if name.startswith("huggingface_hub") and module:
                        for attr in ("tqdm", "tqdm_auto"):
                            if hasattr(module, attr):
                                self.originals[(module, attr)] = getattr(module, attr)
                                try:
                                    setattr(module, attr, self.wrapper)
                                except Exception:
                                    pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                for (target, attr), original in self.originals.items():
                    try:
                        setattr(target, attr, original)
                    except Exception:
                        pass

        if progress:
            progress(0.05, f"Downloading {repo_id}")

        with PatchTqdm(TqdmProgressWrapper):
            snapshot_download(repo_id=repo_id, local_dir=destination, token=token or None, ignore_patterns=IGNORED_WEIGHT_PATTERNS)

        model = self.get(destination.relative_to(self.config.models_dir).as_posix())
        if progress:
            progress(1.0, f"Saved {model.shard_count} Safetensors file(s)")
        return destination
