from __future__ import annotations

import json
from pathlib import Path

import pytest

from gguf_rig.config import RigConfig
from gguf_rig.library import ModelLibrary, normalize_dtype
from gguf_rig.process_manager import ActiveModel, VllmServerManager


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

    assert command[:3] == [str(config.python_executable), "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--dtype") + 1] == "float16"
    assert command[command.index("--load-format") + 1] == "safetensors"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--api-key") + 1] == "secret"
    assert "--enforce-eager" in command
    assert "--enable-chunked-prefill" in command


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
    assert status["auto_restart"] is False


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
vllm_num_requests_running 3
vllm_num_requests_waiting 1
vllm_gpu_cache_usage_perc 0.45
vllm_avg_generation_throughput_toks_per_s 120.5
# Some labeled metric we skip
vllm_request_duration{method="POST"} 0.123
"""
    result = _parse_prometheus_simple(text)
    assert result["vllm_num_requests_running"] == 3.0
    assert result["vllm_num_requests_waiting"] == 1.0
    assert result["vllm_gpu_cache_usage_perc"] == 0.45
    assert result["vllm_avg_generation_throughput_toks_per_s"] == 120.5
    assert "vllm_request_duration" not in result  # Labeled lines are skipped.
