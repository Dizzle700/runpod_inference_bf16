from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from gguf_rig.config import RigConfig
from gguf_rig.chat_client import ChatClient
from gguf_rig.library import ModelLibrary, normalize_dtype
from gguf_rig.process_manager import ActiveModel, VllmServerManager
from gguf_rig.streaming import iter_sse_data


def make_config(tmp_path: Path, **overrides) -> RigConfig:
    values = dict(
        volume_root=tmp_path,
        models_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        python_executable=tmp_path / "python",
        api_host="127.0.0.1",
        api_port=8000,
        panel_host="127.0.0.1",
        panel_port=7860,
        api_key="",
        panel_user="",
        panel_password="",
        hf_token="",
        allow_insecure=False,
        health_timeout=1,
        stop_timeout=1,
        auto_restart=False,
        auto_restart_max_retries=3,
        max_log_bytes=50 * 1024 * 1024,
    )
    values.update(overrides)
    return RigConfig(**values)


def write_safetensors(path: Path, dtype: str = "BF16", payload_size: int = 4) -> None:
    header = json.dumps({"weight": {"dtype": dtype, "shape": [payload_size], "data_offsets": [0, payload_size]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"x" * payload_size)


def make_model(config: RigConfig, model_id: str = "org/repo", dtypes=("BF16",)) -> Path:
    repo = config.models_dir / model_id
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}")
    for index, dtype in enumerate(dtypes, start=1):
        write_safetensors(repo / f"model-{index:05d}-of-{len(dtypes):05d}.safetensors", dtype)
    return repo


@pytest.mark.parametrize(("value", "expected"), [("bf16", "bfloat16"), ("BFLOAT16", "bfloat16"), ("fp16", "float16"), ("half", "float16"), ("fp32", "float32")])
def test_normalize_dtype(value: str, expected: str):
    assert normalize_dtype(value) == expected


def test_library_scans_model_directories_and_reads_headers(tmp_path: Path):
    config = make_config(tmp_path)
    make_model(config, dtypes=("BF16", "F16"))

    records = ModelLibrary(config).scan()

    assert [record.id for record in records] == ["org/repo"]
    assert records[0].dtypes == ("bf16", "fp16")
    assert records[0].shard_count == 2


def test_library_requires_config_and_safetensors(tmp_path: Path):
    config = make_config(tmp_path)
    incomplete = config.models_dir / "org" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}")

    assert ModelLibrary(config).scan() == []
    with pytest.raises(FileNotFoundError, match="Safetensors"):
        ModelLibrary(config).get("org/incomplete")


def test_library_rejects_incomplete_indexed_snapshot(tmp_path: Path):
    config = make_config(tmp_path)
    repo = config.models_dir / "org" / "incomplete"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}")
    write_safetensors(repo / "model-00001-of-00002.safetensors")
    (repo / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        }
    }))

    library = ModelLibrary(config)
    assert library.scan() == []
    with pytest.raises(FileNotFoundError, match="missing weight file"):
        library.get("org/incomplete")


def test_failed_download_preserves_previous_model(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    previous = make_model(config)
    previous_config = (previous / "config.json").read_text()

    def incomplete_download(*, local_dir, **kwargs):
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text('{"new": true}')
        write_safetensors(local_dir / "model-00001-of-00002.safetensors")
        raise RuntimeError("network interrupted")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=incomplete_download),
    )

    with pytest.raises(RuntimeError, match="network interrupted"):
        ModelLibrary(config).download_snapshot("org/repo")

    assert (previous / "config.json").read_text() == previous_config
    assert not list((config.models_dir / ".staging").glob("*/org/repo"))


def test_download_progress_tolerates_tqdm_without_description(tmp_path: Path, monkeypatch):
    """Some tqdm versions report progress before assigning ``desc``."""
    config = make_config(tmp_path)
    reported: list[tuple[float, str]] = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            self.total = 10
            self.n = 0

        def update(self, n=1):
            self.n += n
            return True

    def snapshot_download(*, local_dir, tqdm_class, **kwargs):
        progress = tqdm_class(total=10)
        progress.update(1)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        write_safetensors(local_dir / "model.safetensors")

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.__path__ = []
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "tqdm.auto", types.SimpleNamespace(tqdm=FakeTqdm))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )

    ModelLibrary(config).download_snapshot(
        "org/repo", progress=lambda value, description: reported.append((value, description))
    )

    assert any(description == "Downloading" for _, description in reported)


def test_library_rejects_path_escape(tmp_path: Path):
    library = ModelLibrary(make_config(tmp_path))
    with pytest.raises(ValueError, match="escapes"):
        library.get("../outside")


def test_library_delete_model(tmp_path: Path):
    config = make_config(tmp_path)
    make_model(config)
    lib = ModelLibrary(config)

    assert len(lib.scan()) == 1
    result = lib.delete("org/repo")
    assert "Deleted" in result
    assert len(lib.scan()) == 0
    # Parent org/ directory should also be cleaned up.
    assert not (config.models_dir / "org").exists()


def test_library_delete_rejects_path_escape(tmp_path: Path):
    library = ModelLibrary(make_config(tmp_path))
    with pytest.raises(ValueError, match="escapes"):
        library.delete("../outside")


def test_library_delete_nonexistent(tmp_path: Path):
    library = ModelLibrary(make_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        library.delete("org/nonexistent")


def test_model_operation_lock_serializes_delete(tmp_path: Path):
    config = make_config(tmp_path)
    make_model(config)
    library = ModelLibrary(config)
    completed = threading.Event()

    def delete():
        library.delete("org/repo")
        completed.set()

    with library.operation_lock:
        worker = threading.Thread(target=delete)
        worker.start()
        time.sleep(0.05)
        assert not completed.is_set()
    worker.join(timeout=1)
    assert completed.is_set()


def test_remote_repo_validation():
    assert ModelLibrary.validate_repo_id("org/repo") == "org/repo"
    assert ModelLibrary.validate_repo_id("https://huggingface.co/org/repo/tree/main") == "org/repo"
    assert ModelLibrary.validate_repo_id("hf://org/repo") == "org/repo"
    with pytest.raises(ValueError):
        ModelLibrary.validate_repo_id("https://example.com/org/repo")


def test_public_listeners_require_secrets(tmp_path: Path):
    config = make_config(tmp_path, api_host="0.0.0.0", panel_host="0.0.0.0")
    with pytest.raises(RuntimeError, match="SAFETENSORS_API_KEY"):
        config.validate_security()


def test_runpod_proxy_urls_are_built_from_pod_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123xyz")
    config = make_config(tmp_path, api_port=8000, panel_port=7860)

    assert config.public_api_url == "https://abc123xyz-8000.proxy.runpod.net"
    assert config.public_panel_url == "https://abc123xyz-7860.proxy.runpod.net"


def test_public_url_overrides_take_priority(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123xyz")
    config = make_config(
        tmp_path,
        public_api_url_override="https://api.example.test/",
        public_panel_url_override="https://panel.example.test/",
    )

    assert config.public_api_url == "https://api.example.test"
    assert config.public_panel_url == "https://panel.example.test"


def test_build_command_uses_safetensors_and_selected_dtype(tmp_path: Path):
    config = make_config(tmp_path, api_key="secret")
    config.python_executable.write_bytes(b"python")
    model = make_model(config)
    manager = VllmServerManager(config, ModelLibrary(config))

    command = manager.build_command(ActiveModel(
        model_id="org/repo",
        dtype="fp16",
        tensor_parallel_size=2,
        enforce_eager=True,
        enable_chunked_prefill=True
    ))

    assert command[:2] == [str(config.python_executable), str(Path(__file__).parents[1] / "vllm_launcher.py")]
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--dtype") + 1] == "float16"
    assert command[command.index("--load-format") + 1] == "safetensors"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert json.loads(command[command.index("--limit-mm-per-prompt") + 1]) == {"audio": 1}
    assert command[command.index("--api-key") + 1] == "secret"
    assert "--enforce-eager" in command
    assert "--enable-chunked-prefill" in command


def test_server_log_redacts_api_keys(tmp_path: Path):
    config = make_config(tmp_path, api_key="super-secret-token")
    manager = VllmServerManager(config, ModelLibrary(config))

    line = "non-default args: {'api_key': ['super-secret-token'], 'model': 'local'}"

    assert manager._redact_server_log_line(line) == "non-default args: {'api_key': ['***'], 'model': 'local'}"


def test_preflight_rejects_tensor_parallel_larger_than_visible_gpus(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    make_model(config)
    manager = VllmServerManager(config, ModelLibrary(config))
    monkeypatch.setattr(
        "gguf_rig.process_manager.gpu_stats",
        lambda: [{"memory_total_mib": 24_576}],
    )

    with pytest.raises(ValueError, match="exceeds"):
        manager.preflight(ActiveModel(model_id="org/repo", tensor_parallel_size=2))


def test_context_fallback_halves_context_after_oom(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    make_model(config)
    manager = VllmServerManager(config, ModelLibrary(config))
    attempted = []
    monkeypatch.setattr(manager, "preflight", lambda active: {
        "tensor_parallel_size": 1,
        "estimated_weight_gib_per_gpu": 1.0,
        "available_gib_per_gpu": 20.0,
    })

    def launch_once(active, persist):
        attempted.append(active.max_model_len)
        if len(attempted) == 1:
            raise RuntimeError("CUDA out of memory while allocating KV cache")

    monkeypatch.setattr(manager, "_launch_once", launch_once)
    manager._launch(ActiveModel(model_id="org/repo", max_model_len=8192), persist=False)

    assert attempted == [8192, 4096]


def test_saved_state_round_trip(tmp_path: Path):
    config = make_config(tmp_path)
    make_model(config)
    manager = VllmServerManager(config, ModelLibrary(config))
    active = ActiveModel(
        model_id="org/repo",
        dtype="float32",
        max_model_len=4096,
        enforce_eager=True,
        enable_chunked_prefill=True
    )

    manager._write_state(active)

    assert manager.load_saved() == active
    payload = json.loads(config.active_model_file.read_text())
    assert payload["schema_version"] == 2
    assert "api_key" not in payload
    assert "hf_token" not in payload
    assert payload["enforce_eager"] is True
    assert payload["enable_chunked_prefill"] is True


def test_config_new_fields_defaults(tmp_path: Path):
    config = make_config(tmp_path)
    assert config.auto_restart is False
    assert config.auto_restart_max_retries == 3
    assert config.max_log_bytes == 50 * 1024 * 1024
    assert config.panel_find_free_port is False


def test_config_new_fields_custom(tmp_path: Path):
    config = make_config(tmp_path, auto_restart=True, auto_restart_max_retries=5, max_log_bytes=100 * 1024 * 1024)
    assert config.auto_restart is True
    assert config.auto_restart_max_retries == 5
    assert config.max_log_bytes == 100 * 1024 * 1024


def test_api_stats(tmp_path: Path):
    config = make_config(tmp_path)
    manager = VllmServerManager(config, ModelLibrary(config))

    assert manager.api_stats()["total_requests"] == 0
    manager.record_api_call(tokens=10, latency=0.5, error=False)
    manager.record_api_call(tokens=20, latency=1.5, error=True)

    stats = manager.api_stats()
    assert stats["total_requests"] == 2
    assert stats["total_tokens"] == 30
    assert stats["errors"] == 1
    assert stats["avg_latency_s"] == 1.0


def test_status_includes_model_params(tmp_path: Path):
    config = make_config(tmp_path)
    manager = VllmServerManager(config, ModelLibrary(config))

    status = manager.status()
    # When no model is active, extended params are None.
    assert status["max_model_len"] is None
    assert status["max_audio_per_prompt"] is None
    assert status["auto_restart"] is False


def test_audio_upload_is_encoded_as_vllm_data_url(tmp_path: Path):
    config = make_config(tmp_path)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFFtest-audio")
    client = ChatClient(config, VllmServerManager(config, ModelLibrary(config)))

    data_url = client._audio_data_url(str(audio))

    assert data_url == "data:audio/wav;base64,UklGRnRlc3QtYXVkaW8="


def test_audio_upload_respects_size_limit(tmp_path: Path):
    config = make_config(tmp_path, max_audio_upload_bytes=4)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFFtest")
    client = ChatClient(config, VllmServerManager(config, ModelLibrary(config)))

    with pytest.raises(ValueError, match="limit"):
        client._audio_data_url(str(audio))


def test_log_rotation(tmp_path: Path):
    config = make_config(tmp_path, max_log_bytes=100)
    config.ensure_directories()
    manager = VllmServerManager(config, ModelLibrary(config))

    # Create a log file exceeding the limit.
    config.server_log_file.write_text("x" * 200)
    assert config.server_log_file.stat().st_size > 100

    manager._rotate_log_if_needed()

    # Original should be gone, rotated file should exist.
    assert not config.server_log_file.exists()
    rotated = config.server_log_file.with_suffix(".log.1")
    assert rotated.exists()
    assert rotated.stat().st_size == 200


def test_prometheus_parser():
    from gguf_rig.process_manager import _parse_prometheus_simple

    text = """
# HELP vllm_num_requests_running Number of running requests
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model_name="current",engine="0"} 2
vllm_num_requests_running{model_name="current",engine="1"} 1
vllm_num_requests_waiting{model_name="current"} 1
vllm_gpu_cache_usage_perc 0.45
vllm_avg_generation_throughput_toks_per_s 120.5
vllm_request_duration{method="POST"} 0.123
"""
    result = _parse_prometheus_simple(text)
    assert result["vllm_num_requests_running"] == 3.0
    assert result["vllm_num_requests_waiting"] == 1.0
    assert result["vllm_gpu_cache_usage_perc"] == 0.45
    assert result["vllm_avg_generation_throughput_toks_per_s"] == 120.5
    assert result["vllm_request_duration"] == 0.123


def test_sse_parser_stops_at_done_without_consuming_more_data():
    def lines():
        yield b': keep-alive\n'
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
        yield b'\n'
        yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n'
        yield b'\n'
        yield b'data: [DONE]\n'
        yield b'\n'
        raise AssertionError("SSE parser consumed data after [DONE]")

    assert list(iter_sse_data(lines())) == [
        '{"choices":[{"delta":{"content":"hello"}}]}',
        '{"choices":[],"usage":{"completion_tokens":1}}',
    ]


def test_sse_parser_accepts_multiline_data_and_eof_without_blank_line():
    assert list(iter_sse_data([
        b"data: first\n",
        b"data:second\n",
    ])) == ["first\nsecond"]
