from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .config import RigConfig


REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
DTYPE_NAMES = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}
IGNORED_WEIGHT_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack", "*.onnx", "*.gguf"]

# Safetensors header limit: 10 MB is generous; legitimate headers are typically < 2 MB.
_MAX_HEADER_BYTES = 10 * 1024 * 1024
_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<number>\d{5})-of-(?P<total>\d{5})\.safetensors$")
_METADATA_FILE = ".rig-metadata.json"
_COMPLETE_FILE = ".download-complete"


class DownloadCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRecord:
    id: str
    path: Path
    size_bytes: int
    dtypes: tuple[str, ...]
    shard_count: int
    revision: str = ""
    commit_sha: str = ""
    status: str = "ready"

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
    revision: str
    commit_sha: str

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
            if not 2 <= header_length <= _MAX_HEADER_BYTES:
                return set()
            header = json.loads(handle.read(header_length))
        return {
            DTYPE_NAMES.get(str(value.get("dtype", "")).upper(), str(value.get("dtype", "")).lower())
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict) and value.get("dtype")
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return set()


def _validate_model_snapshot(model_dir: Path) -> list[Path]:
    """Return verified root weight files, rejecting incomplete snapshots."""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in: {model_dir}")

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No Safetensors weights found in: {model_dir}")

    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    for index_path in indexes:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is empty")
            expected_names = {str(name) for name in weight_map.values()}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Safetensors index {index_path.name}: {exc}") from exc

        for name in expected_names:
            expected = (model_dir / name).resolve()
            if model_dir.resolve() not in expected.parents or not expected.is_file():
                raise FileNotFoundError(f"Incomplete snapshot: missing weight file {name}")

    # Some repositories omit the index despite using conventional numbered shards.
    groups: dict[tuple[str, int], set[int]] = {}
    for shard in shards:
        match = _SHARD_RE.match(shard.name)
        if match:
            key = (match.group("prefix"), int(match.group("total")))
            groups.setdefault(key, set()).add(int(match.group("number")))
    for (prefix, total), present in groups.items():
        expected_numbers = set(range(1, total + 1))
        if present != expected_numbers:
            missing = sorted(expected_numbers - present)
            raise FileNotFoundError(
                f"Incomplete snapshot: {prefix} is missing shard(s) "
                + ", ".join(f"{number:05d}-of-{total:05d}" for number in missing[:10])
            )
    return shards


def _read_metadata(model_dir: Path) -> dict[str, str]:
    try:
        payload = json.loads((model_dir / _METADATA_FILE).read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload.items()}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


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
        self.operation_lock = threading.RLock()
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
            if ".cache" in model_dir.parts or ".staging" in model_dir.parts:
                continue
            try:
                shards = _validate_model_snapshot(model_dir)
            except (FileNotFoundError, ValueError):
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
                revision=_read_metadata(model_dir).get("revision", ""),
                commit_sha=_read_metadata(model_dir).get("commit_sha", ""),
            ))
        return sorted(records, key=lambda item: item.id.lower())

    def get(self, model_id: str) -> ModelRecord:
        model_dir = self._safe_local_path(self.config.models_dir / model_id)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model not found: {model_id}")
        shards = _validate_model_snapshot(model_dir)
        dtypes: set[str] = set()
        for shard in shards:
            dtypes.update(_read_safetensors_dtypes(shard))
        metadata = _read_metadata(model_dir)
        return ModelRecord(
            model_dir.relative_to(self.config.models_dir.resolve()).as_posix(),
            model_dir,
            sum(p.stat().st_size for p in shards),
            tuple(sorted(dtypes)),
            len(shards),
            metadata.get("revision", ""),
            metadata.get("commit_sha", ""),
        )

    def delete(self, model_id: str) -> str:
        """Delete a downloaded model from the persistent volume."""
        with self.operation_lock:
            return self._delete_unlocked(model_id)

    def _delete_unlocked(self, model_id: str) -> str:
        model_dir = self._safe_local_path(self.config.models_dir / model_id)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model not found: {model_id}")
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_gib = size_bytes / 1024**3
        shutil.rmtree(model_dir)
        # Clean up empty parent directories (e.g. org/ after deleting org/repo).
        for parent in model_dir.parents:
            if parent == self.config.models_dir.resolve() or parent == self.config.models_dir:
                break
            try:
                parent.rmdir()  # Only succeeds if empty.
            except OSError:
                break
        return f"Deleted {model_id} ({size_gib:.2f} GiB freed)"

    def inspect_remote(self, repo_id: str, token: str | None = None, revision: str = "") -> RemoteModel:
        repo_id = self.validate_repo_id(repo_id)
        revision = revision.strip() or "main"
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is not installed") from exc
        info = HfApi(token=token or None).model_info(
            repo_id=repo_id, revision=revision, files_metadata=True
        )
        tensors = [item for item in info.siblings or [] if getattr(item, "rfilename", "").lower().endswith(".safetensors")]
        if not tensors:
            raise ValueError("This repository has no .safetensors weights")
        sizes = [getattr(item, "size", None) for item in tensors]
        total = sum(int(size) for size in sizes if size is not None) if all(
            size is not None for size in sizes
        ) else None
        config_dtype = "unknown"
        safetensors_meta = getattr(info, "safetensors", None)
        parameters = getattr(safetensors_meta, "parameters", None)
        if isinstance(parameters, dict) and parameters:
            config_dtype = "/".join(sorted(str(key).lower() for key in parameters))
        return RemoteModel(
            repo_id, total, len(tensors), config_dtype, revision, str(getattr(info, "sha", "") or "")
        )

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.config.models_dir).free

    def download_snapshot(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        revision: str = "main",
        commit_sha: str = "",
        expected_size: int | None = None,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Path:
        with self.operation_lock:
            return self._download_snapshot_unlocked(
                repo_id,
                token=token,
                revision=revision,
                commit_sha=commit_sha,
                expected_size=expected_size,
                progress=progress,
                cancelled=cancelled,
            )

    def _download_snapshot_unlocked(
        self,
        repo_id: str,
        *,
        token: str | None,
        revision: str,
        commit_sha: str,
        expected_size: int | None,
        progress: Callable[[float, str], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> Path:
        repo_id = self.validate_repo_id(repo_id)
        if expected_size is not None and self.free_bytes() < expected_size + 1024**3:
            raise OSError("Not enough free volume space (a 1 GiB safety margin is required)")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is not installed") from exc
        destination = self.config.models_dir.joinpath(*repo_id.split("/"))
        staging_root = self.config.models_dir / ".staging" / uuid.uuid4().hex
        staging = staging_root.joinpath(*repo_id.split("/"))
        backup = staging_root / ".previous"
        staging_root.mkdir(parents=True, exist_ok=True)
        (staging_root / "status.json").write_text(
            json.dumps({"repo_id": repo_id, "revision": revision, "status": "downloading"}),
            encoding="utf-8",
        )

        from tqdm.auto import tqdm

        progress_lock = threading.Lock()

        class ProgressTqdm(tqdm):
            """Per-download progress adapter; avoids process-global monkey-patching."""

            def __init__(self, *args, **kwargs):
                kwargs.setdefault("disable", True)
                super().__init__(*args, **kwargs)
                self._report()

            def update(self, n=1):
                updated = super().update(n)
                self._report()
                return updated

            def _report(self):
                if cancelled and cancelled():
                    raise DownloadCancelled(f"Download cancelled: {repo_id}@{revision}")
                if not progress:
                    return
                total = float(self.total or 0)
                ratio = min(0.95, max(0.05, float(self.n) / total)) if total else 0.05
                # tqdm can call this hook while its own constructor is still
                # initializing, before ``desc`` has been assigned.
                description = str(getattr(self, "desc", None) or "Downloading")
                with progress_lock:
                    progress(ratio, description)

        if progress:
            progress(0.05, f"Downloading {repo_id}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=staging,
                revision=commit_sha or revision,
                token=token or None,
                ignore_patterns=IGNORED_WEIGHT_PATTERNS,
                tqdm_class=ProgressTqdm,
            )
            if cancelled and cancelled():
                raise DownloadCancelled(f"Download cancelled: {repo_id}@{revision}")
            verified_shards = _validate_model_snapshot(staging)
            (staging / _METADATA_FILE).write_text(
                json.dumps(
                    {
                        "repo_id": repo_id,
                        "revision": revision,
                        "commit_sha": commit_sha,
                        "downloaded_at": int(time.time()),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (staging / _COMPLETE_FILE).write_text("ok\n", encoding="utf-8")

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.rename(backup)
            try:
                staging.rename(destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            if progress:
                progress(1.0, f"Saved {len(verified_shards)} Safetensors file(s)")
            return destination
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def statuses(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        records = self.scan()
        ready_ids = {record.id for record in records}
        for record in records:
            result.append({
                "model": record.id,
                "revision": record.revision or "unknown",
                "commit": record.commit_sha[:12] or "unknown",
                "status": "ready",
            })
        root = self.config.models_dir.resolve()
        for config_path in root.rglob("config.json"):
            if ".cache" in config_path.parts or ".staging" in config_path.parts:
                continue
            model_id = config_path.parent.relative_to(root).as_posix()
            if model_id not in ready_ids:
                try:
                    _validate_model_snapshot(config_path.parent)
                    status = "ready"
                except FileNotFoundError:
                    status = "incomplete"
                except ValueError:
                    status = "broken"
                result.append({
                    "model": model_id,
                    "revision": "unknown",
                    "commit": "unknown",
                    "status": status,
                })
        for status_path in (root / ".staging").glob("*/status.json"):
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                result.append({
                    "model": str(payload.get("repo_id", "unknown")),
                    "revision": str(payload.get("revision", "unknown")),
                    "commit": "pending",
                    "status": "downloading",
                })
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return result
